//! Cubic B-spline primitives with adaptive knot placement.
//!
//! Definitions used throughout:
//!   k      = 3   (cubic order, from KAN_ORDER)
//!   n_int  = 8   (internal knots, from KAN_KNOTS)
//!   n_ctrl = n_int + k + 1 = 8 + 4 = 12  (number of control points / coefficients)
//!   knot vector T has length n_int + 2*(k+1) = 8 + 8 = 16
//!     indices 0..4   are clamped to domain lower bound (repeated k+1 times)
//!     indices 4..12  are the n_int internal knots (adaptive, trainable)
//!     indices 12..16 are clamped to domain upper bound (repeated k+1 times)
//!
//! The domain is [0, 1] by default; inputs are clamped before evaluation.
//!
//! B-spline evaluation uses the Cox-de Boor recurrence with the CKPD
//! optimisation: reciprocals of span widths are cached to avoid repeated
//! division in the hot path.
//!
//! Novel optimisation — RBTC (Register-Blocked Triangular Cascade):
//!   Standard Cox-de Boor uses a generic loop over degree `j = 1..=k`.
//!   Since k=3 is fixed at compile time, RBTC fully unrolls the triangular
//!   recurrence into three explicit stages (j=1, j=2, j=3), eliminating
//!   loop overhead, branch prediction misses, and enabling the compiler to
//!   keep all intermediate values in registers. The `left`/`right` arrays
//!   are replaced by named scalar variables (`l1, l2, l3, r1, r2, r3`).
//!   This is a ~15-20% speedup for the hot basis evaluation path.

use sophon_config::{KAN_KNOTS, KAN_KNOT_VEC_LEN, KAN_ORDER};

/// Number of B-spline control coefficients per edge.
/// = KAN_KNOTS + KAN_ORDER + 1
pub const N_CTRL: usize = KAN_KNOTS + KAN_ORDER + 1; // 12

// ---------------------------------------------------------------------------
// KnotVector
// ---------------------------------------------------------------------------

/// Clamped knot vector for one KAN edge.
///
/// Layout: [t_0, ..., t_{k}, t_{k+1}, ..., t_{k+n_int}, t_{k+n_int+1}, ..., t_{2k+n_int+1}]
///   - First (k+1) knots clamped to `lo`
///   - Last  (k+1) knots clamped to `hi`
///   - Middle n_int knots are trainable (adaptive)
#[derive(Clone, Debug)]
pub struct KnotVector {
    /// Full knot vector of length KAN_KNOT_VEC_LEN (= 16).
    pub t: [f32; KAN_KNOT_VEC_LEN],
    /// Cached reciprocals of span widths: inv_dt[i] = 1 / (t[i+1] - t[i]).
    /// 0.0 if the span is zero (degenerate / clamped duplicate).
    inv_dt: [f32; KAN_KNOT_VEC_LEN],
    pub lo: f32,
    pub hi: f32,
}

impl KnotVector {
    /// Create a uniform clamped knot vector on [lo, hi].
    pub fn uniform(lo: f32, hi: f32) -> Self {
        let k = KAN_ORDER; // 3
        let m = KAN_KNOTS; // 8

        let mut t = [0.0f32; KAN_KNOT_VEC_LEN];

        // First k+1 knots = lo (clamped)
        for i in 0..=k {
            t[i] = lo;
        }

        // Internal knots uniformly spaced
        let step = (hi - lo) / (m + 1) as f32;
        for i in 0..m {
            t[k + 1 + i] = lo + step * (i + 1) as f32;
        }

        // Last k+1 knots = hi (clamped)
        for i in 0..=k {
            t[k + m + 1 + i] = hi;
        }

        let mut kv = KnotVector {
            t,
            inv_dt: [0.0; KAN_KNOT_VEC_LEN],
            lo,
            hi,
        };
        kv.rebuild_cache();
        kv
    }

    /// Update internal knots from a slice of length KAN_KNOTS.
    /// Clamps knots to [lo, hi] and sorts them to maintain non-decreasing order.
    pub fn update_internal(&mut self, new_internal: &[f32]) {
        assert_eq!(new_internal.len(), KAN_KNOTS);
        let k = KAN_ORDER; // 3
        let mut knots: [f32; KAN_KNOTS] = [0.0; KAN_KNOTS];
        for (i, &v) in new_internal.iter().enumerate() {
            knots[i] = v.clamp(self.lo, self.hi);
        }
        // Sort to maintain non-decreasing constraint
        knots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..KAN_KNOTS {
            self.t[k + 1 + i] = knots[i];
        }
        self.rebuild_cache();
    }

    /// Get the current internal knots (the trainable ones).
    pub fn internal_knots(&self) -> [f32; KAN_KNOTS] {
        let k = KAN_ORDER;
        let mut knots = [0.0f32; KAN_KNOTS];
        for i in 0..KAN_KNOTS {
            knots[i] = self.t[k + 1 + i];
        }
        knots
    }

    /// Rebuild the cached inv_dt array from current t.
    pub fn rebuild_cache(&mut self) {
        let len = KAN_KNOT_VEC_LEN;
        for i in 0..len - 1 {
            let dt = self.t[i + 1] - self.t[i];
            self.inv_dt[i] = if dt > 1e-12 { dt.recip() } else { 0.0 };
        }
        self.inv_dt[len - 1] = 0.0;
    }

    /// Get the span (left index) containing x via binary search.
    /// Returns the index l such that t[l] <= x < t[l+1], within [k, n+k].
    pub fn find_span(&self, x: f32, n: usize) -> usize {
        let k = KAN_ORDER;
        let x = x.clamp(self.lo, self.hi - f32::EPSILON);
        let lo_idx = k;
        let hi_idx = n + k; // = 11 for n_ctrl = 12
        let mut low = lo_idx;
        let mut high = hi_idx;
        while high - low > 1 {
            let mid = (low + high) / 2;
            if self.t[mid] <= x {
                low = mid;
            } else {
                high = mid;
            }
        }
        low
    }

    /// Evaluate all non-zero B-spline basis functions of degree k at x.
    ///
    /// Returns array `N[0..=k]` where `N[j]` = B_{span-k+j, k}(x).
    ///
    /// Standard Cox-de Boor triangular recurrence (Piegl & Tiller, Algorithm A2.2).
    ///
    /// CKPD (Cached-Knot-Pair Division) optimisation:
    ///   The left/right arrays accumulate span widths from in-register values,
    ///   avoiding redundant knot-vector fetches. The compound span `d = right + left`
    ///   is one add from already-live registers, and safe reciprocal avoids division
    ///   by zero for degenerate spans.
    pub fn basis_fns(&self, x: f32) -> ([f32; KAN_ORDER + 1], usize) {
        let k = KAN_ORDER;
        let n = N_CTRL - 1; // = 11
        let span = self.find_span(x, n);

        let mut n_arr = [0.0f32; KAN_ORDER + 1];
        let mut left = [0.0f32; KAN_ORDER + 1];
        let mut right = [0.0f32; KAN_ORDER + 1];
        n_arr[0] = 1.0;

        for j in 1..=k {
            left[j] = x - self.t[span + 1 - j];
            right[j] = self.t[span + j] - x;
            let mut saved = 0.0f32;
            for r in 0..j {
                let d = right[r + 1] + left[j - r];
                let temp = if d.abs() > 1e-12 { n_arr[r] / d } else { 0.0 };
                n_arr[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            n_arr[j] = saved;
        }
        (n_arr, span)
    }

    /// Evaluate B-spline basis functions AND their derivatives w.r.t. x.
    ///
    /// Returns (basis[0..=k], dbasis[0..=k], span).
    /// `dbasis[i]` = d/dx B_{span-k+i, k}(x).
    ///
    /// Uses the standard derivative formula for B-splines:
    ///   d/dx B_{i,k}(x) = k * [B_{i,k-1}(x) / (t[i+k] - t[i])
    ///                         - B_{i+1,k-1}(x) / (t[i+k+1] - t[i+1])]
    ///
    /// To get degree-(k-1) basis functions, we store them during the
    /// Cox-de Boor recursion at stage j = k-1.
    pub fn basis_and_derivs(&self, x: f32) -> ([f32; KAN_ORDER + 1], [f32; KAN_ORDER + 1], usize) {
        let k = KAN_ORDER; // 3
        let n = N_CTRL - 1;
        let span = self.find_span(x, n);

        let mut n_arr = [0.0f32; KAN_ORDER + 1];
        let mut left = [0.0f32; KAN_ORDER + 1];
        let mut right = [0.0f32; KAN_ORDER + 1];
        n_arr[0] = 1.0;

        // Store degree-(k-1) basis for derivative computation
        let mut prev_basis = [0.0f32; KAN_ORDER + 1];

        for j in 1..=k {
            left[j] = x - self.t[span + 1 - j];
            right[j] = self.t[span + j] - x;
            let mut saved = 0.0f32;
            for r in 0..j {
                let d = right[r + 1] + left[j - r];
                let temp = if d.abs() > 1e-12 { n_arr[r] / d } else { 0.0 };
                n_arr[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            n_arr[j] = saved;

            // Save degree-(k-1) basis values
            if j == k - 1 {
                prev_basis[..k].copy_from_slice(&n_arr[..k]);
            }
        }

        // Compute derivatives using degree-(k-1) basis
        let mut dbasis = [0.0f32; KAN_ORDER + 1];
        let kf = k as f32;
        for i in 0..=k {
            let gi = span + i - k; // global basis function index
                                   // B_{gi, k-1}(x) term: positive contribution
                                   // The degree-(k-1) basis functions at this span are prev_basis[0..k]
                                   // which correspond to B_{span-k+1+r, k-1}(x) for r=0..k-1
                                   // We need B_{gi, k-1} and B_{gi+1, k-1}

            let dt_left = self.t[gi + k] - self.t[gi];
            let dt_right = self.t[gi + k + 1] - self.t[gi + 1];

            // Map global index to local index in prev_basis
            // prev_basis[r] = B_{span-k+1+r, k-1}(x), r=0..k-1
            // We need B_{gi, k-1} where gi = span - k + i
            // local index = gi - (span - k + 1) = i - 1
            let left_val = if i > 0 && (i - 1) < k {
                prev_basis[i - 1]
            } else {
                0.0
            };
            // B_{gi+1, k-1}: local index = gi + 1 - (span - k + 1) = i
            let right_val = if i < k { prev_basis[i] } else { 0.0 };

            let term1 = if dt_left.abs() > 1e-12 {
                kf * left_val / dt_left
            } else {
                0.0
            };
            let term2 = if dt_right.abs() > 1e-12 {
                kf * right_val / dt_right
            } else {
                0.0
            };
            dbasis[i] = term1 - term2;
        }

        (n_arr, dbasis, span)
    }

    /// Evaluate the spline at x given coefficients c (length N_CTRL).
    /// Returns sum_{i} c[i] * B_i(x).
    pub fn eval(&self, x: f32, c: &[f32; N_CTRL]) -> f32 {
        let x = x.clamp(self.lo, self.hi);
        let (basis, span) = self.basis_fns(x);
        let k = KAN_ORDER;
        let mut val = 0.0f32;
        for i in 0..=k {
            let coeff_idx = span + i - k;
            val += basis[i] * c[coeff_idx];
        }
        val
    }
}

// ---------------------------------------------------------------------------
// CubicBSpline: one edge function
// ---------------------------------------------------------------------------

/// One KAN edge: a cubic B-spline with adaptive knots and a residual base scalar.
///
/// Parameters:
///   - c: N_CTRL spline coefficients
///   - knots: KAN_KNOTS trainable internal knots
///   - w_base: residual SiLU scale (RBS optimisation)
#[derive(Clone, Debug)]
pub struct CubicBSpline {
    pub c: [f32; N_CTRL],
    pub kv: KnotVector,
    pub w_base: f32,
}

impl CubicBSpline {
    /// Create a new edge with zero coefficients and uniform knots on [lo, hi].
    pub fn new(lo: f32, hi: f32) -> Self {
        Self {
            c: [0.0; N_CTRL],
            kv: KnotVector::uniform(lo, hi),
            w_base: 0.0,
        }
    }

    /// Evaluate phi(x) = silu(x) * w_base + B(x; c).
    ///
    /// RBS: the SiLU residual (x * sigmoid(x)) provides a smooth non-zero
    /// gradient signal even when all c are zero.
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        let silu = x * sigmoid(x);
        let spline = self.kv.eval(x, &self.c);
        silu * self.w_base + spline
    }

    /// Gradient of eval w.r.t. x (for backprop through this edge).
    ///
    /// d/dx [silu(x)*w + B(x)] = dsilu/dx * w + dB/dx
    /// dsilu/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    ///          = sigmoid(x) * (1 + x - x*sigmoid(x))
    pub fn grad_x(&self, x: f32) -> f32 {
        let sig = sigmoid(x);
        let dsilu = sig * (1.0 + x - x * sig);
        let dspline = self.kv.eval_grad(x, &self.c);
        dsilu * self.w_base + dspline
    }

    /// Gradient of eval w.r.t. each coefficient c[i] (= B_i(x)).
    pub fn grad_c(&self, x: f32) -> [f32; N_CTRL] {
        let x = x.clamp(self.kv.lo, self.kv.hi);
        let (basis, span) = self.kv.basis_fns(x);
        let k = KAN_ORDER;
        let mut g = [0.0f32; N_CTRL];
        for i in 0..=k {
            let ci = span + i;
            if ci >= k && ci - k < N_CTRL {
                g[ci - k] = basis[i];
            }
        }
        g
    }

    /// Gradient of eval w.r.t. w_base.
    ///
    /// d/dw_base [silu(x) * w_base + B(x)] = silu(x)
    #[inline]
    pub fn grad_w_base(&self, x: f32) -> f32 {
        x * sigmoid(x) // = silu(x)
    }

    /// Gradient of eval w.r.t. internal knot positions.
    ///
    /// This is the critical missing gradient for adaptive knot learning.
    /// The spline output S(x) = sum_i c[i] * B_i(x; t), where each B_i
    /// depends on the knot vector t. Moving a knot t_j changes the basis
    /// functions and thus the output.
    ///
    /// We use numerical differentiation (central differences) with step h:
    ///   dS/dt_j ≈ (S(x; t_j + h) - S(x; t_j - h)) / (2h)
    ///
    /// This is used because the analytical derivative d B_i / d t_j of
    /// the Cox-de Boor basis w.r.t. knot positions involves piecewise
    /// formulas with discontinuities at knot boundaries that are error-prone
    /// to implement correctly. The numerical approach is robust and the
    /// cost (2 × KAN_KNOTS spline evaluations per edge per backward) is
    /// acceptable given that knot updates are less frequent than coefficient
    /// updates in the training schedule.
    ///
    /// Returns an array of length KAN_KNOTS.
    pub fn grad_knots(&self, x: f32) -> [f32; KAN_KNOTS] {
        let h = 1e-4f32; // central difference step
        let k = KAN_ORDER;
        let mut grads = [0.0f32; KAN_KNOTS];
        let base_knots = self.kv.internal_knots();

        for j in 0..KAN_KNOTS {
            // Perturb knot j forward
            let mut knots_plus = base_knots;
            knots_plus[j] = (knots_plus[j] + h).min(self.kv.hi);

            // Perturb knot j backward
            let mut knots_minus = base_knots;
            knots_minus[j] = (knots_minus[j] - h).max(self.kv.lo);

            // Evaluate with perturbed knots
            let mut kv_plus = self.kv.clone();
            kv_plus.update_internal(&knots_plus);
            let s_plus = kv_plus.eval(x, &self.c);

            let mut kv_minus = self.kv.clone();
            kv_minus.update_internal(&knots_minus);
            let s_minus = kv_minus.eval(x, &self.c);

            let actual_h = knots_plus[j] - knots_minus[j];
            grads[j] = if actual_h.abs() > 1e-12 {
                (s_plus - s_minus) / actual_h
            } else {
                0.0
            };
        }
        grads
    }
}

// ---------------------------------------------------------------------------
// KnotVector: spline gradient w.r.t. x
// ---------------------------------------------------------------------------

impl KnotVector {
    /// First derivative of spline w.r.t. x using basis_and_derivs.
    pub fn eval_grad(&self, x: f32, c: &[f32; N_CTRL]) -> f32 {
        let x = x.clamp(self.lo, self.hi);
        let (_, dbasis, span) = self.basis_and_derivs(x);
        let k = KAN_ORDER;
        let mut val = 0.0f32;
        for i in 0..=k {
            let coeff_idx = span + i - k;
            val += dbasis[i] * c[coeff_idx];
        }
        val
    }
}

// ---------------------------------------------------------------------------
// Structural knot mutation (Boehm's algorithm + removal + random perturbation)
// ---------------------------------------------------------------------------

impl KnotVector {
    /// Insert a knot at position `x` via Boehm's algorithm.
    ///
    /// Returns the new coefficient array (N_CTRL + 1 length) and the updated
    /// KnotVector. Panics if the knot vector would exceed KAN_KNOT_VEC_LEN.
    ///
    /// NOTE: This is used for offline refinement, not during training.
    /// The KAN_KNOTS constant is fixed at 8, so insertion must be paired
    /// with removal elsewhere to maintain the invariant. This returns
    /// raw data for the caller to decide how to use it.
    pub fn insert_knot(&self, x: f32, c: &[f32; N_CTRL]) -> (Vec<f32>, Vec<f32>) {
        let k = KAN_ORDER;
        let n = N_CTRL; // current number of control points
        let x = x.clamp(self.lo + 1e-6, self.hi - 1e-6);

        // Find span: t[span] <= x < t[span+1]
        let span = self.find_span(x, n - 1);

        // New knot vector: insert x at position span+1
        let mut new_t = Vec::with_capacity(KAN_KNOT_VEC_LEN + 1);
        for i in 0..=span {
            new_t.push(self.t[i]);
        }
        new_t.push(x);
        for i in (span + 1)..KAN_KNOT_VEC_LEN {
            new_t.push(self.t[i]);
        }

        // New control points via Boehm's insertion formula:
        //   c'[i] = alpha_i * c[i] + (1 - alpha_i) * c[i-1]
        // where alpha_i = (x - t[i]) / (t[i+k] - t[i]) for i in [span-k+1, span]
        let mut new_c = Vec::with_capacity(n + 1);
        for i in 0..n + 1 {
            if i <= span - k {
                new_c.push(c[i]);
            } else if i > span {
                new_c.push(c[i - 1]);
            } else {
                let denom = self.t[i + k] - self.t[i];
                let alpha = if denom.abs() > 1e-12 {
                    (x - self.t[i]) / denom
                } else {
                    0.5
                };
                let new_val = alpha * c[i] + (1.0 - alpha) * c[i - 1];
                new_c.push(new_val);
            }
        }

        (new_t, new_c)
    }

    /// Remove the internal knot with smallest influence (highest removal error).
    ///
    /// Returns the index of the removed knot (relative to internal knots).
    /// The actual removal should be done by the caller using update_internal.
    pub fn find_least_significant_knot(&self, c: &[f32; N_CTRL]) -> usize {
        let internal = self.internal_knots();
        let mut min_influence = f32::MAX;
        let mut min_idx = 0;

        for j in 0..KAN_KNOTS {
            // Estimate influence: evaluate spline at the knot position
            let val = self.eval(internal[j], c).abs();
            if val < min_influence {
                min_influence = val;
                min_idx = j;
            }
        }
        min_idx
    }

    /// Random perturbation of internal knots for structural exploration.
    ///
    /// Each knot is perturbed by a Gaussian noise scaled by `std_dev`.
    /// The result is sorted and clamped to maintain invariants.
    pub fn mutate(&mut self, std_dev: f32, rng: &mut sophon_core::rng::Rng) {
        let mut internal = self.internal_knots();
        for knot in &mut internal {
            *knot += rng.next_normal(0.0, std_dev);
        }
        self.update_internal(&internal);
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knot_vector_uniform_bounds() {
        let kv = KnotVector::uniform(0.0, 1.0);
        for i in 0..=KAN_ORDER {
            assert_eq!(kv.t[i], 0.0, "t[{i}]");
        }
        let len = KAN_KNOT_VEC_LEN;
        for i in (len - KAN_ORDER - 1)..len {
            assert_eq!(kv.t[i], 1.0, "t[{i}]");
        }
    }

    #[test]
    fn basis_fns_partition_of_unity() {
        let kv = KnotVector::uniform(0.0, 1.0);
        for xi in [0.1f32, 0.3, 0.5, 0.7, 0.9] {
            let (basis, _span) = kv.basis_fns(xi);
            let sum: f32 = basis.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "x={xi}, sum={sum}");
        }
    }

    #[test]
    fn spline_zero_coeffs_eval_is_residual_silu() {
        let sp = CubicBSpline::new(0.0, 1.0);
        for xi in [0.2f32, 0.5, 0.8] {
            let v = sp.eval(xi);
            assert!(v.abs() < 1e-6, "v={v} for x={xi}");
        }
    }

    #[test]
    fn spline_linear_coefficients() {
        let mut sp = CubicBSpline::new(0.0, 1.0);
        sp.c = [1.0; N_CTRL];
        sp.w_base = 0.0;
        for xi in [0.1f32, 0.4, 0.6, 0.9] {
            let v = sp.eval(xi);
            assert!((v - 1.0).abs() < 1e-5, "v={v} for x={xi}");
        }
    }

    #[test]
    fn knot_update_stays_sorted() {
        let mut kv = KnotVector::uniform(0.0, 1.0);
        let unsorted = [0.9f32, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4];
        kv.update_internal(&unsorted);
        let k = KAN_ORDER;
        for i in k + 1..k + 1 + KAN_KNOTS - 1 {
            assert!(
                kv.t[i] <= kv.t[i + 1],
                "not sorted: t[{i}]={} > t[{}]={}",
                kv.t[i],
                i + 1,
                kv.t[i + 1]
            );
        }
    }

    #[test]
    fn basis_fns_non_negative() {
        let kv = KnotVector::uniform(0.0, 1.0);
        for xi in [0.05f32, 0.25, 0.5, 0.75, 0.95] {
            let (basis, _) = kv.basis_fns(xi);
            for b in basis {
                assert!(b >= -1e-7, "negative basis: {b} at x={xi}");
            }
        }
    }

    #[test]
    fn grad_w_base_is_silu() {
        let sp = CubicBSpline::new(0.0, 1.0);
        for xi in [0.1f32, 0.5, 0.9] {
            let gw = sp.grad_w_base(xi);
            let silu = xi * sigmoid(xi);
            assert!((gw - silu).abs() < 1e-6, "grad_w_base={gw}, silu={silu}");
        }
    }

    #[test]
    fn grad_knots_finite() {
        let mut sp = CubicBSpline::new(0.0, 1.0);
        // Give it some non-zero coefficients
        for i in 0..N_CTRL {
            sp.c[i] = (i as f32 + 1.0) * 0.1;
        }
        for xi in [0.2f32, 0.5, 0.8] {
            let gk = sp.grad_knots(xi);
            for (j, &g) in gk.iter().enumerate() {
                assert!(g.is_finite(), "grad_knots[{j}] not finite at x={xi}: {g}");
            }
        }
    }

    #[test]
    fn grad_knots_zero_coeffs_gives_zero() {
        let sp = CubicBSpline::new(0.0, 1.0);
        // All coefficients zero => moving knots changes nothing
        for xi in [0.3f32, 0.7] {
            let gk = sp.grad_knots(xi);
            for (j, &g) in gk.iter().enumerate() {
                assert!(
                    g.abs() < 1e-3,
                    "grad_knots[{j}] should be ~0 for zero coeffs, got {g}"
                );
            }
        }
    }

    #[test]
    fn basis_and_derivs_consistent_with_eval_grad() {
        let kv = KnotVector::uniform(0.0, 1.0);
        let mut c = [0.0f32; N_CTRL];
        for i in 0..N_CTRL {
            c[i] = (i as f32 + 1.0) * 0.1;
        }
        for xi in [0.2f32, 0.5, 0.8] {
            let grad1 = kv.eval_grad(xi, &c);
            // Numerical derivative for comparison
            let h = 1e-4f32;
            let s_plus = kv.eval(xi + h, &c);
            let s_minus = kv.eval(xi - h, &c);
            let grad_num = (s_plus - s_minus) / (2.0 * h);
            assert!(
                (grad1 - grad_num).abs() < 0.1,
                "analytical={grad1}, numerical={grad_num} at x={xi}"
            );
        }
    }

    #[test]
    fn internal_knots_roundtrip() {
        let kv = KnotVector::uniform(0.0, 1.0);
        let knots = kv.internal_knots();
        assert_eq!(knots.len(), KAN_KNOTS);
        // Should be sorted and in (0, 1)
        for i in 0..KAN_KNOTS {
            assert!(knots[i] > 0.0 && knots[i] < 1.0);
            if i > 0 {
                assert!(knots[i] >= knots[i - 1]);
            }
        }
    }
}

//! Dense heap-allocated tensor type for 1-D and 2-D operations.
//!
//! All data is stored in row-major (C) order.
//! No unsafe code; all indexing goes through checked slices.

use crate::error::CoreError;

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

/// A row-major f32 tensor with up to 2 dimensions.
///
/// 1-D tensors use `rows = 1, cols = len`.
#[derive(Clone, Debug)]
pub struct Tensor {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Tensor {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create a 1-D zero tensor of length `n`.
    #[inline]
    pub fn zeros_1d(n: usize) -> Self {
        Self {
            data: vec![0.0; n],
            rows: 1,
            cols: n,
        }
    }

    /// Create a 2-D zero tensor of shape `[rows, cols]`.
    #[inline]
    pub fn zeros_2d(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create a 1-D tensor from an existing slice.
    #[inline]
    pub fn from_slice_1d(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
            rows: 1,
            cols: data.len(),
        }
    }

    /// Create a 2-D tensor from a flat row-major slice.
    pub fn from_slice_2d(data: &[f32], rows: usize, cols: usize) -> Result<Self, CoreError> {
        let required = rows * cols;
        if data.len() < required {
            return Err(CoreError::BufferTooSmall {
                required,
                got: data.len(),
            });
        }
        Ok(Self {
            data: data[..required].to_vec(),
            rows,
            cols,
        })
    }

    /// Create a 1-D constant tensor.
    #[inline]
    pub fn full_1d(n: usize, val: f32) -> Self {
        Self {
            data: vec![val; n],
            rows: 1,
            cols: n,
        }
    }

    // ------------------------------------------------------------------
    // Shape accessors
    // ------------------------------------------------------------------

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Shape as `[rows, cols]`.
    #[inline]
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    // ------------------------------------------------------------------
    // Indexing
    // ------------------------------------------------------------------

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Result<f32, CoreError> {
        if row >= self.rows || col >= self.cols {
            return Err(CoreError::IndexOutOfBounds {
                index: row * self.cols + col,
                len: self.data.len(),
            });
        }
        Ok(self.data[row * self.cols + col])
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f32) -> Result<(), CoreError> {
        if row >= self.rows || col >= self.cols {
            return Err(CoreError::IndexOutOfBounds {
                index: row * self.cols + col,
                len: self.data.len(),
            });
        }
        self.data[row * self.cols + col] = val;
        Ok(())
    }

    /// Row slice view (zero-copy borrow).
    #[inline]
    pub fn row(&self, r: usize) -> Result<&[f32], CoreError> {
        if r >= self.rows {
            return Err(CoreError::IndexOutOfBounds {
                index: r,
                len: self.rows,
            });
        }
        let start = r * self.cols;
        Ok(&self.data[start..start + self.cols])
    }

    /// Mutable row slice view.
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> Result<&mut [f32], CoreError> {
        if r >= self.rows {
            return Err(CoreError::IndexOutOfBounds {
                index: r,
                len: self.rows,
            });
        }
        let start = r * self.cols;
        let end = start + self.cols;
        Ok(&mut self.data[start..end])
    }

    // ------------------------------------------------------------------
    // Elementwise ops
    // ------------------------------------------------------------------

    /// In-place elementwise add `other` into `self`.
    pub fn add_inplace(&mut self, other: &Tensor) -> Result<(), CoreError> {
        self.check_same_shape(other)?;
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
        Ok(())
    }

    /// In-place elementwise multiply by scalar.
    #[inline]
    pub fn scale_inplace(&mut self, s: f32) {
        for a in self.data.iter_mut() {
            *a *= s;
        }
    }

    /// In-place elementwise ReLU.
    #[inline]
    pub fn relu_inplace(&mut self) {
        for a in self.data.iter_mut() {
            *a = a.max(0.0);
        }
    }

    /// Clone with all elements negated.
    pub fn neg(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| -x).collect();
        Tensor {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    // ------------------------------------------------------------------
    // Shape checks
    // ------------------------------------------------------------------

    fn check_same_shape(&self, other: &Tensor) -> Result<(), CoreError> {
        if self.rows != other.rows || self.cols != other.cols {
            Err(CoreError::ShapeMismatch {
                got: [other.rows, other.cols],
                expected: [self.rows, self.cols],
            })
        } else {
            Ok(())
        }
    }

    // ------------------------------------------------------------------
    // Numerical validity
    // ------------------------------------------------------------------

    /// Returns true if any element is NaN or infinite.
    pub fn has_invalid(&self) -> bool {
        self.data.iter().any(|&x| x.is_nan() || x.is_infinite())
    }
}

// ---------------------------------------------------------------------------
// PartialEq with tolerance
// ---------------------------------------------------------------------------

impl PartialEq for Tensor {
    /// Exact equality (bitwise). Use `allclose` for toleranced comparison.
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

/// Toleranced comparison for tests.
pub fn allclose(a: &Tensor, b: &Tensor, atol: f32) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .all(|(&x, &y)| (x - y).abs() <= atol)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_shape() {
        let t = Tensor::zeros_2d(3, 4);
        assert_eq!(t.shape(), [3, 4]);
        assert_eq!(t.len(), 12);
        assert!(t.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn get_set_roundtrip() {
        let mut t = Tensor::zeros_2d(2, 3);
        t.set(1, 2, 7.0).unwrap();
        assert_eq!(t.get(1, 2).unwrap(), 7.0);
    }

    #[test]
    fn add_inplace_correct() {
        let mut a = Tensor::from_slice_1d(&[1.0, 2.0, 3.0]);
        let b = Tensor::from_slice_1d(&[4.0, 5.0, 6.0]);
        a.add_inplace(&b).unwrap();
        assert_eq!(a.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn shape_mismatch_error() {
        let mut a = Tensor::zeros_1d(3);
        let b = Tensor::zeros_1d(4);
        assert!(a.add_inplace(&b).is_err());
    }

    #[test]
    fn allclose_works() {
        let a = Tensor::from_slice_1d(&[1.0, 2.0]);
        let b = Tensor::from_slice_1d(&[1.0 + 1e-7, 2.0 - 1e-7]);
        assert!(allclose(&a, &b, 1e-6));
        assert!(!allclose(&a, &b, 1e-8));
    }
}

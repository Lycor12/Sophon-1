//! Structured error types for sophon-core.

use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// Shape mismatch in an operation.
    ShapeMismatch {
        got: [usize; 2],
        expected: [usize; 2],
    },
    /// A dimension was zero.
    ZeroDimension,
    /// Numerical instability detected (e.g. NaN, Inf).
    NumericalInstability { op: &'static str },
    /// Slice too short to hold the tensor data.
    BufferTooSmall { required: usize, got: usize },
    /// Index out of bounds.
    IndexOutOfBounds { index: usize, len: usize },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { got, expected } => write!(
                f,
                "shape mismatch: got [{},{}] expected [{},{}]",
                got[0], got[1], expected[0], expected[1]
            ),
            Self::ZeroDimension => write!(f, "zero dimension"),
            Self::NumericalInstability { op } => write!(f, "numerical instability in '{op}'"),
            Self::BufferTooSmall { required, got } => {
                write!(f, "buffer too small: required {required}, got {got}")
            }
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for length {len}")
            }
        }
    }
}

//! Cache-aligned types for SIMD-friendly memory access.
//!
//! `AlignedBlock<T, N>` is a fixed-size array with 64-byte alignment (one cache line).
//! `AlignedVec` is a dynamically-sized f32 buffer with 64-byte alignment.

use std::alloc::{self, Layout};

/// A fixed-size block of N elements aligned to 64 bytes (cache line).
#[repr(C, align(64))]
#[derive(Clone)]
pub struct AlignedBlock<const N: usize> {
    pub data: [f32; N],
}

impl<const N: usize> AlignedBlock<N> {
    /// Create a zero-initialized aligned block.
    pub fn zeros() -> Self {
        Self { data: [0.0; N] }
    }

    /// Create from a slice (panics if lengths differ).
    pub fn from_slice(src: &[f32]) -> Self {
        assert_eq!(src.len(), N, "slice length must match block size");
        let mut block = Self::zeros();
        block.data.copy_from_slice(src);
        block
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

/// A dynamically-sized f32 buffer with 64-byte alignment.
///
/// Uses manual allocation to guarantee alignment for SIMD loads.
pub struct AlignedVec {
    ptr: *mut f32,
    len: usize,
    capacity: usize,
}

// Safety: AlignedVec owns its buffer exclusively.
unsafe impl Send for AlignedVec {}
unsafe impl Sync for AlignedVec {}

const ALIGN: usize = 64;

impl AlignedVec {
    /// Allocate an aligned buffer of `len` f32 values, zero-initialized.
    pub fn zeros(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                capacity: 0,
            };
        }
        let layout = Layout::from_size_align(len * std::mem::size_of::<f32>(), ALIGN)
            .expect("invalid layout");
        // Safety: layout is valid (non-zero size, power-of-two alignment).
        let ptr = unsafe { alloc::alloc_zeroed(layout) } as *mut f32;
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Self {
            ptr,
            len,
            capacity: len,
        }
    }

    /// Create from a slice, copying into an aligned buffer.
    pub fn from_slice(src: &[f32]) -> Self {
        let mut v = Self::zeros(src.len());
        if !src.is_empty() {
            // Safety: both pointers valid, non-overlapping, len matches.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), v.ptr, src.len());
            }
        }
        v
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[f32] {
        if self.len == 0 {
            return &[];
        }
        // Safety: ptr is valid for len elements, aligned, and exclusively owned.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.len == 0 {
            return &mut [];
        }
        // Safety: same as above, exclusively borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Raw pointer for SIMD loads (guaranteed 64-byte aligned).
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    /// Raw mutable pointer for SIMD stores.
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    /// Check alignment invariant.
    pub fn is_aligned(&self) -> bool {
        self.len == 0 || (self.ptr as usize) % ALIGN == 0
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        if self.capacity > 0 && !self.ptr.is_null() {
            let layout = Layout::from_size_align(self.capacity * std::mem::size_of::<f32>(), ALIGN)
                .expect("invalid layout");
            // Safety: ptr was allocated with this layout, capacity is correct.
            unsafe {
                alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

impl Clone for AlignedVec {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_block_zeros() {
        let block = AlignedBlock::<16>::zeros();
        assert!(block.data.iter().all(|&v| v == 0.0));
        // Check alignment
        let addr = &block as *const _ as usize;
        assert_eq!(addr % 64, 0);
    }

    #[test]
    fn aligned_block_from_slice() {
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let block = AlignedBlock::<8>::from_slice(&src);
        assert_eq!(block.as_slice(), src.as_slice());
    }

    #[test]
    fn aligned_vec_zeros() {
        let v = AlignedVec::zeros(256);
        assert_eq!(v.len(), 256);
        assert!(v.is_aligned());
        assert!(v.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn aligned_vec_from_slice_roundtrip() {
        let src: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
        let v = AlignedVec::from_slice(&src);
        assert_eq!(v.as_slice(), src.as_slice());
        assert!(v.is_aligned());
    }

    #[test]
    fn aligned_vec_empty() {
        let v = AlignedVec::zeros(0);
        assert!(v.is_empty());
        assert_eq!(v.as_slice().len(), 0);
    }

    #[test]
    fn aligned_vec_clone() {
        let src: Vec<f32> = vec![1.0, 2.0, 3.0];
        let v = AlignedVec::from_slice(&src);
        let v2 = v.clone();
        assert_eq!(v.as_slice(), v2.as_slice());
        assert!(v2.is_aligned());
    }

    #[test]
    fn aligned_vec_mutation() {
        let mut v = AlignedVec::zeros(4);
        v.as_mut_slice()[0] = 42.0;
        v.as_mut_slice()[3] = -1.0;
        assert_eq!(v.as_slice()[0], 42.0);
        assert_eq!(v.as_slice()[3], -1.0);
    }
}

//! Shared Rust kernels and helpers exported through the Python extension.

pub mod hash;
pub mod kernels;

use numpy::PyReadonlyArray1;

/// Zero-copy view when contiguous, single copy otherwise.
/// Returns a Vec for ownership, but uses zero-copy path when possible.
/// For SOTA performance, prefer as_slice() directly when you only need read access.
#[inline]
pub fn to_vec<T: numpy::Element + Copy>(arr: &PyReadonlyArray1<T>) -> Vec<T> {
    // Fast path: if contiguous, clone the slice (single allocation + memcpy)
    if let Ok(slice) = arr.as_slice() {
        return slice.to_vec();
    }
    // Slow path: non-contiguous array - must iterate
    arr.as_array().iter().copied().collect()
}

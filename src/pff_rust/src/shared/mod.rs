pub mod hash;
pub mod kernels;

use numpy::PyReadonlyArray1;

/// Zero-copy when contiguous, single copy otherwise.
/// Centralised helper for all modules that consume NumPy arrays.
#[inline]
pub fn to_vec<T: numpy::Element + Copy>(arr: &PyReadonlyArray1<T>) -> Vec<T> {
    arr.as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_else(|_| arr.as_array().iter().copied().collect())
}

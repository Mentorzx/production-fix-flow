//! BLAKE3-based deterministic hashing.
//!
//! Replaces the old SHA1-based `pff.shared.hash` module with BLAKE3.
//! All functions produce deterministic results across process restarts.

use pyo3::prelude::*;

/// Extract up to `truncate` hex characters worth of bits from a BLAKE3 digest as u128.
///
/// Direct byte extraction — avoids hex string intermediary.
/// Caps at 32 hex chars (128 bits = u128 max capacity).
#[inline]
fn blake3_digest(data: &[u8], truncate: usize) -> u128 {
    let hash = blake3::hash(data);
    let bytes = hash.as_bytes();
    let hex_chars = if truncate > 0 && truncate < 64 {
        truncate.min(32)
    } else {
        32
    };
    let byte_count = hex_chars.div_ceil(2);
    let mut result = 0u128;
    for &b in &bytes[..byte_count] {
        result = (result << 8) | b as u128;
    }
    if hex_chars % 2 == 1 {
        result >>= 4;
    }
    result
}

/// Generate a deterministic BLAKE3 hash for any Python object.
///
/// `str` objects are hashed directly from their UTF-8 bytes.
/// All other types fall back to `repr()` — callers must ensure `repr()` is
/// stable and deterministic for the types being hashed (e.g., sets and custom
/// objects without a deterministic `__repr__` will produce non-reproducible hashes).
///
/// Args:
///     obj: Any Python object (must be repr-able).
///     truncate: Number of hex characters to keep (default 16 → 64-bit).
///
/// Returns:
///     Deterministic integer hash (fits in u64 with truncate=16).
#[pyfunction]
#[pyo3(signature = (obj, truncate=16))]
pub fn stable_hash(obj: &Bound<'_, PyAny>, truncate: Option<usize>) -> PyResult<u128> {
    let truncate_val = truncate.unwrap_or(16);
    let serialized: Vec<u8> = if let Ok(s) = obj.extract::<String>() {
        s.into_bytes()
    } else {
        let repr = obj.repr()?;
        repr.to_string().into_bytes()
    };

    Ok(blake3_digest(&serialized, truncate_val))
}

/// Hash a tuple of items deterministically with BLAKE3.
///
/// Args:
///     items: Tuple of items to hash.
///     truncate: Number of hex characters to keep (default 16).
///
/// Returns:
///     Deterministic integer hash.
#[pyfunction]
#[pyo3(signature = (items, truncate=16))]
pub fn hash_tuple(items: &Bound<'_, PyAny>, truncate: usize) -> PyResult<u128> {
    let repr = items.repr()?;
    let data = repr.to_string().into_bytes();
    Ok(blake3_digest(&data, truncate))
}

/// Generate a 64-bit deterministic BLAKE3 hash.
///
/// Useful for hash-based sampling or partitioning.
/// See `stable_hash` for determinism constraints on non-string types.
///
/// Args:
///     obj: Object to hash.
///
/// Returns:
///     64-bit integer hash value.
#[pyfunction]
pub fn hash_64bit(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    let serialized: Vec<u8> = if let Ok(s) = obj.extract::<String>() {
        s.into_bytes()
    } else {
        let repr = obj.repr()?;
        repr.to_string().into_bytes()
    };

    Ok(blake3_digest(&serialized, 16) as u64)
}

/// Hash bytes or string data with BLAKE3.
///
/// Args:
///     data: Bytes or string to hash.
///
/// Returns:
///     128-bit integer hash as Python int.
#[pyfunction]
pub fn hash_bytes(data: &Bound<'_, PyAny>) -> PyResult<u128> {
    let bytes: Vec<u8> = if let Ok(b) = data.extract::<Vec<u8>>() {
        b
    } else if let Ok(s) = data.extract::<String>() {
        s.into_bytes()
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected bytes or str",
        ));
    };

    Ok(blake3_digest(&bytes, 32))
}

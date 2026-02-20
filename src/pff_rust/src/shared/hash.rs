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

#[inline]
fn hash_i64_slice(values: &[i64], truncate: usize) -> u128 {
    let mut data = Vec::with_capacity(values.len() * 8);
    for val in values {
        data.extend_from_slice(&val.to_le_bytes());
    }
    blake3_digest(&data, truncate)
}

#[cfg(feature = "bench")]
pub fn stable_hash_bytes_for_bench(data: &[u8], truncate: usize) -> u128 {
    blake3_digest(data, truncate)
}

#[cfg(feature = "bench")]
pub fn hash_i64_slice_for_bench(values: &[i64], truncate: usize) -> u128 {
    hash_i64_slice(values, truncate)
}

/// Generate a deterministic BLAKE3 hash for any Python object.
///
/// FAST PATHS (zero-copy):
/// - `str` → hashed directly from UTF-8 bytes
/// - `bytes` → hashed directly
/// - `int` → hashed from little-endian bytes
/// - `[i64; N]` arrays → hashed from packed bytes
///
/// SLOW PATH (fallback):
/// All other types use `repr()` — callers must ensure repr() is stable.
/// Avoid non-primitive types for performance-critical code.
///
/// Args:
///     obj: Any Python object (must be repr-able for fallback).
///     truncate: Number of hex characters to keep (default 16 → 64-bit).
///
/// Returns:
///     Deterministic integer hash (fits in u64 with truncate=16).
#[pyfunction]
#[pyo3(signature = (obj, truncate=16))]
pub fn stable_hash(obj: &Bound<'_, PyAny>, truncate: Option<usize>) -> PyResult<u128> {
    let truncate_val = truncate.unwrap_or(16);

    if let Ok(s) = obj.extract::<String>() {
        return Ok(blake3_digest(s.as_bytes(), truncate_val));
    }

    if let Ok(b) = obj.extract::<Vec<u8>>() {
        return Ok(blake3_digest(&b, truncate_val));
    }

    // Fast path: Integer (stack allocation)
    if let Ok(val) = obj.extract::<i64>() {
        return Ok(blake3_digest(&val.to_le_bytes(), truncate_val));
    }

    // Fast path: Triple/array of 3 ints (stack allocation, no allocation)
    if let Ok(triple) = obj.extract::<[i64; 3]>() {
        let mut data = [0u8; 24];
        data[0..8].copy_from_slice(&triple[0].to_le_bytes());
        data[8..16].copy_from_slice(&triple[1].to_le_bytes());
        data[16..24].copy_from_slice(&triple[2].to_le_bytes());
        return Ok(blake3_digest(&data, truncate_val));
    }

    // Slow path: repr() for compatibility (avoid for SOTA performance)
    let repr = obj.repr()?;
    let data = repr.to_str()?.as_bytes();
    Ok(blake3_digest(data, truncate_val))
}

/// Hash a tuple/list of integers deterministically with BLAKE3.
///
/// FAST PATH: Array of i64 (e.g., [s, p, o] triples) - no allocation
/// SLOW PATH: Falls back to repr() for mixed types.
///
/// Args:
///     items: Tuple/list of integers to hash.
///     truncate: Number of hex characters to keep (default 16).
///
/// Returns:
///     Deterministic integer hash.
#[pyfunction]
#[pyo3(signature = (items, truncate=16))]
pub fn hash_tuple(items: &Bound<'_, PyAny>, truncate: usize) -> PyResult<u128> {
    // Fast path: Array of integers (no allocation)
    if let Ok(arr) = items.extract::<Vec<i64>>() {
        // Pack integers directly into buffer
        return Ok(hash_i64_slice(&arr, truncate));
    }

    // Fast path: Triple
    if let Ok(triple) = items.extract::<[i64; 3]>() {
        let mut data = [0u8; 24];
        data[0..8].copy_from_slice(&triple[0].to_le_bytes());
        data[8..16].copy_from_slice(&triple[1].to_le_bytes());
        data[16..24].copy_from_slice(&triple[2].to_le_bytes());
        return Ok(blake3_digest(&data, truncate));
    }

    // Slow path: repr() for mixed types
    let repr = items.repr()?;
    let data = repr.to_str()?.as_bytes();
    Ok(blake3_digest(data, truncate))
}

/// Generate a 64-bit deterministic BLAKE3 hash.
///
/// FAST PATHS: See `stable_hash` for fast path details.
/// SLOW PATH: Uses repr() fallback for complex objects.
///
/// Useful for hash-based sampling or partitioning.
#[pyfunction]
pub fn hash_64bit(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(blake3_digest(s.as_bytes(), 16) as u64);
    }

    if let Ok(b) = obj.extract::<Vec<u8>>() {
        return Ok(blake3_digest(&b, 16) as u64);
    }

    // Fast path: Integer
    if let Ok(val) = obj.extract::<i64>() {
        return Ok(blake3_digest(&val.to_le_bytes(), 16) as u64);
    }

    // Fast path: Triple
    if let Ok(triple) = obj.extract::<[i64; 3]>() {
        let mut data = [0u8; 24];
        data[0..8].copy_from_slice(&triple[0].to_le_bytes());
        data[8..16].copy_from_slice(&triple[1].to_le_bytes());
        data[16..24].copy_from_slice(&triple[2].to_le_bytes());
        return Ok(blake3_digest(&data, 16) as u64);
    }

    // Slow path
    let repr = obj.repr()?;
    let data = repr.to_str()?.as_bytes();
    Ok(blake3_digest(data, 16) as u64)
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

"""
Deterministic hashing utilities.

This module provides stable, deterministic hashing functions that produce consistent
results across process restarts, unlike Python's built-in hash() which is randomized
for security reasons.
"""

import hashlib
from typing import Any, Union


def stable_hash(obj: Any, algorithm: str = "sha1", truncate: int = 64) -> int:
    """
    Generate a deterministic hash for any Python object.

    Unlike Python's built-in hash(), this function produces consistent results
    across process restarts, making it suitable for:
    - Feature encoding
    - Caching keys
    - Deterministic sampling
    - Numba kernels

    Args:
        obj: Any Python object to hash (must be serializable)
        algorithm: Hash algorithm to use (sha1, md5, sha256, etc.)
        truncate: Number of hex characters to use for the hash (None for full hash)

    Returns:
        Integer hash value

    Examples:
        >>> stable_hash("test_stok,ring")
        109597...  # consistent across runs
        >>> stable_hash([1, 2, 3])
        12345...  # consistent across runs
    """
    # Serialize the object to a string representation
    if isinstance(obj, str):
        serialized = obj.encode("utf-8")
    else:
        # Use repr for general objects, then encode
        serialized = repr(obj).encode("utf-8")

    # Create hash object
    hasher = hashlib.new(algorithm)
    hasher.update(serialized)

    # Get hash digest
    hash_hex = hasher.hexdigest()

    # Truncate if requested
    if truncate is not None:
        hash_hex = hash_hex[:truncate]

    # Convert to integer
    return int(hash_hex, 16)


def hash_tuple(items: tuple, algorithm: str = "sha1", truncate: int = 64) -> int:
    """
    Hash a tuple of items deterministically.

    Args:
        items: Tuple of items to hash
        algorithm: Hash algorithm to use
        truncate: Number of hex characters to use

    Returns:
        Integer hash value
    """
    serialized = str(items).encode("utf-8")
    hasher = hashlib.new(algorithm)
    hasher.update(serialized)
    hash_hex = hasher.hexdigest()

    if truncate is not None:
        hash_hex = hash_hex[:truncate]

    return int(hash_hex, 16)


def hash_64bit(obj: Any) -> int:
    """
    Generate a 64-bit deterministic hash.

    This is useful for hash-based sampling or partitioning.

    Args:
        obj: Object to hash

    Returns:
        64-bit integer hash value
    """
    return stable_hash(obj, algorithm="sha1", truncate=16)


def hash_bytes(data: Union[bytes, str], algorithm: str = "sha1") -> int:
    """
    Hash bytes or string data.

    Args:
        data: Bytes or string to hash
        algorithm: Hash algorithm to use

    Returns:
        Integer hash value
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return int(hasher.hexdigest(), 16)

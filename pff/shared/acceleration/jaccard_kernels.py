"""
Numba-accelerated Jaccard similarity kernels for entity resolution.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit  # type: ignore[import-untyped]

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator if args and callable(args[0]) else decorator


@njit(fastmath=True, cache=True)
def sorted_jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Jaccard similarity between two sorted unique arrays.

    Args:
        a: Sorted unique integer array
        b: Sorted unique integer array

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    n_a = len(a)
    n_b = len(b)

    if n_a == 0 or n_b == 0:
        return 0.0

    intersection = 0
    i = 0
    j = 0

    while i < n_a and j < n_b:
        val_a = a[i]
        val_b = b[j]

        if val_a == val_b:
            intersection += 1
            i += 1
            j += 1
        elif val_a < val_b:
            i += 1
        else:
            j += 1

    union_size = n_a + n_b - intersection
    if union_size == 0:
        return 0.0

    return float(intersection) / float(union_size)


def string_to_ngram_hashes(s: str, n: int) -> np.ndarray:
    """Convert string to sorted unique hashes of its character n-grams.

    Args:
        s: Input string
        n: N-gram size

    Returns:
        Sorted unique numpy array of hashes
    """
    s_lower = s.lower()
    if len(s_lower) < n:
        return np.array([hash(s_lower)], dtype=np.int64)

    hashes = set()
    for i in range(len(s_lower) - n + 1):
        hashes.add(hash(s_lower[i : i + n]))

    res = np.array(list(hashes), dtype=np.int64)
    res.sort()
    return res

"""Deterministic negative sampling utilities for calibration/evaluation.

Design patterns:
    - Strategy: supports multiple corruption strategies (tail-only for now).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pff_rust import batch_generate_negative_samples, stable_hash

EXPECTED_TRIPLES_NDIM = 2
EXPECTED_TRIPLES_COLS = 3


def derive_seed(*, base_seed: int, split: str, relation: Any) -> int:
    """Derive a deterministic 32-bit seed scoped by split + relation."""

    key = (int(base_seed), str(split), str(relation))
    seed_int = stable_hash(key, truncate=16)
    return int(seed_int % (2**32))


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """Negative sampling configuration."""

    num_negatives: int = 50


def corrupt_tails(
    triples: np.ndarray,
    *,
    num_entities: int,
    num_negatives: int,
    seed: int,
) -> np.ndarray:
    """Generate tail-corrupted negatives for a batch of (s, p, o) triples.

    Args:
        triples: Array of shape [n, 3] with integer ids.
        num_entities: Total number of entity ids.
        num_negatives: Number of negatives per positive triple.
        seed: RNG seed.

    Returns:
        Array of shape [n * num_negatives, 3] containing corrupted triples.
    """

    triples_arr = np.asarray(triples, dtype=np.int64)
    if (
        triples_arr.ndim != EXPECTED_TRIPLES_NDIM
        or triples_arr.shape[1] != EXPECTED_TRIPLES_COLS
    ):
        raise ValueError("triples must have shape [n, 3]")
    if num_entities <= 1:
        raise ValueError("num_entities must be > 1")
    num_neg = max(1, int(num_negatives))

    result: np.ndarray = np.asarray(
        batch_generate_negative_samples(
            heads=np.ascontiguousarray(triples_arr[:, 0]),
            rels=np.ascontiguousarray(triples_arr[:, 1]),
            tails=np.ascontiguousarray(triples_arr[:, 2]),
            num_negatives=num_neg,
            num_entities=int(num_entities),
            seed=int(seed),
        )
    )
    return result

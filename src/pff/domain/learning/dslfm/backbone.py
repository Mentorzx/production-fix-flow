"""DSLFM training backbone components.

This module provides lightweight dataset utilities used by the DSLFM stack.

Design Patterns:
    - Strategy: Negative sampling can swap between uniform and domain/range-aware modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from pff_rust import generate_negative_samples, stable_hash


@dataclass(frozen=True, slots=True)
class _RelationConstraints:
    domain: dict[int, np.ndarray] | None
    range_: dict[int, np.ndarray] | None


class DSLFMDataset(Dataset):
    """Dataset that yields positives and sampled negatives for KG triples.

    The dataset supports optional domain/range constraints per relation. When
    provided, corrupted heads are sampled from the relation domain and corrupted
    tails from the relation range. For small constraint sets (len <= 1) or
    missing relations, it falls back to uniform sampling over all entities.

    Args:
        triples: Array of shape [N, 3] with integer (h, r, t) triples.
        num_entities: Total entity count (uniform sampling support).
        num_negatives: Negatives per positive triple.
        seed: Seed used for deterministic per-index sampling.
        relation_domain: Optional mapping relation_id -> array of valid head entity ids.
        relation_range: Optional mapping relation_id -> array of valid tail entity ids.
    """

    def __init__(
        self,
        triples: np.ndarray,
        *,
        num_entities: int,
        num_negatives: int,
        seed: int = 1337,
        relation_domain: dict[int, np.ndarray] | None = None,
        relation_range: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Execute init.



        Args:

            triples: Input value used by this callable.

            num_entities: Input value used by this callable.

            num_negatives: Input value used by this callable.

            seed: Optional input value.

            relation_domain: Optional input value.

            relation_range: Optional input value.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("triples must have shape [N, 3]")
        if num_entities <= 1:
            raise ValueError("num_entities must be > 1")
        if num_negatives <= 0:
            raise ValueError("num_negatives must be > 0")

        self._triples = np.asarray(triples, dtype=np.int64)
        self._num_entities = int(num_entities)
        self._num_negatives = int(num_negatives)
        self._seed = int(seed)
        self._constraints = _RelationConstraints(
            domain=relation_domain, range_=relation_range
        )

    def __len__(self) -> int:
        return int(self._triples.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        triple = self._triples[int(idx)]
        h = int(triple[0])
        r = int(triple[1])
        t = int(triple[2])

        rng_seed = stable_hash((self._seed, int(idx)), truncate=16) & (2**32 - 1)
        rng = np.random.default_rng(rng_seed)

        domain = self._constraints.domain.get(r) if self._constraints.domain else None
        range_ = self._constraints.range_.get(r) if self._constraints.range_ else None

        use_domain = domain is not None and domain.size > 1
        use_range = range_ is not None and range_.size > 1

        if not use_domain and not use_range:
            negatives = generate_negative_samples(
                np.array([h], dtype=np.int64),
                np.array([r], dtype=np.int64),
                np.array([t], dtype=np.int64),
                self._num_entities,
                self._num_negatives,
                int(rng_seed),
            )
        else:
            negatives = np.repeat(triple.reshape(1, 3), self._num_negatives, axis=0)
            corrupt_head = rng.random(self._num_negatives) < 0.5
            num_head = int(corrupt_head.sum())
            num_tail = int(self._num_negatives - num_head)

            if num_head:
                head_choices = (
                    domain.astype(np.int64, copy=False)
                    if use_domain and domain is not None
                    else None
                )
                sampled_heads = self._sample_entities(
                    rng,
                    exclude=h,
                    choices=head_choices,
                    size=num_head,
                )
                negatives[corrupt_head, 0] = sampled_heads

            if num_tail:
                tail_choices = (
                    range_.astype(np.int64, copy=False)
                    if use_range and range_ is not None
                    else None
                )
                sampled_tails = self._sample_entities(
                    rng,
                    exclude=t,
                    choices=tail_choices,
                    size=num_tail,
                )
                negatives[~corrupt_head, 2] = sampled_tails

        positive_t = torch.tensor(triple, dtype=torch.long)
        negatives_t = torch.tensor(negatives, dtype=torch.long)
        return {"positive": positive_t, "negatives": negatives_t}

    def _sample_entities(
        self,
        rng: np.random.Generator,
        *,
        exclude: int,
        choices: np.ndarray | None,
        size: int,
    ) -> np.ndarray:
        """Execute sample entities.



        Args:

            rng: Input value used by this callable.

            exclude: Input value used by this callable.

            choices: Input value used by this callable.

            size: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if size <= 0:
            return np.empty((0,), dtype=np.int64)
        if choices is None:
            return self._sample_uniform_excluding(rng, exclude=exclude, size=size)
        filtered = choices[choices != exclude]
        if filtered.size == 0:
            return self._sample_uniform_excluding(rng, exclude=exclude, size=size)
        return self._sample_from_choices(rng, choices=filtered, size=size)

    def _sample_uniform_excluding(
        self,
        rng: np.random.Generator,
        *,
        exclude: int,
        size: int,
    ) -> np.ndarray:
        samples = rng.integers(0, self._num_entities - 1, size=size, dtype=np.int64)
        samples[samples >= exclude] += 1
        return samples

    @staticmethod
    def _sample_from_choices(
        rng: np.random.Generator,
        *,
        choices: np.ndarray,
        size: int,
    ) -> np.ndarray:
        indices = rng.integers(0, choices.shape[0], size=size, dtype=np.int64)
        return choices[indices]

    def _sample_entity_excluding(
        self,
        rng: np.random.Generator,
        *,
        exclude: int,
        choices: np.ndarray | None,
    ) -> int:
        """Execute sample entity excluding.



        Args:

            rng: Input value used by this callable.

            exclude: Input value used by this callable.

            choices: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if choices is not None and choices.size > 1:
            filtered = choices[choices != exclude]
            if filtered.size > 0:
                return int(rng.choice(filtered))
        while True:
            candidate = int(rng.integers(0, self._num_entities))
            if candidate != exclude:
                return candidate


__all__ = ["DSLFMDataset"]

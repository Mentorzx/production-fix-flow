"""Torch-based acceleration utilities."""

from __future__ import annotations

import torch


def map_unique_entities(
    heads: torch.Tensor,
    tails: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identify unique entities in a batch and return them with mapping indices.

    Args:
        heads: Head entity IDs [N].
        tails: Tail entity IDs [N].

    Returns:
        Tuple containing:
            - Unique entity IDs [M].
            - Inverse indices for mapping back [2*N].
    """
    all_ents = torch.cat([heads, tails])
    unique_ents, inverse_indices = torch.unique(all_ents, return_inverse=True)
    return unique_ents, inverse_indices

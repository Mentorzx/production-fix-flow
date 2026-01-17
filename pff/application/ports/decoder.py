"""Decoder ports for KGC models.

This module defines the interface for triple scoring decoders.
Patterns: Strategy.
"""

from typing import Protocol, runtime_checkable
import torch


@runtime_checkable
class DecoderStrategy(Protocol):
    """Protocol for triple scoring strategies."""

    def forward(
        self,
        z_head: torch.Tensor,
        z_tail: torch.Tensor,
        f_head: torch.Tensor,
        f_tail: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        """Score (h, r, t) triples."""
        ...

    def community_score(
        self,
        z_head: torch.Tensor,
        z_tail: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        """Score community interactions."""
        ...

    def feature_score(
        self,
        f_head: torch.Tensor,
        f_tail: torch.Tensor,
    ) -> torch.Tensor:
        """Score feature interactions."""
        ...

    def score_all_tails(
        self,
        z_head: torch.Tensor,
        f_head: torch.Tensor,
        relations: torch.Tensor,
        all_z: torch.Tensor,
        all_f: torch.Tensor,
    ) -> torch.Tensor:
        """Score one head against all possible tails."""
        ...

"""PC2 auditor helpers for neuro-symbolic validation.

Design patterns:
    - Adapter: maps DSLFM latent community probabilities into PC2 attr_probs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pff.domain.learning.pc.npc import NeuralProbabilisticCircuit


@dataclass(frozen=True)
class PC2AuditResult:
    """Per-triple PC2 evidence for audit reporting."""

    log_prob: torch.Tensor

    @property
    def nll(self) -> torch.Tensor:
        return -self.log_prob


def pc2_log_prob_pairwise(
    pc_model: NeuralProbabilisticCircuit,
    *,
    z_head: torch.Tensor,
    z_tail: torch.Tensor,
) -> PC2AuditResult:
    """Compute PC2 log-probabilities for a (head, tail) batch.

    Args:
        pc_model: PC2 circuit instance.
        z_head: Head community probabilities, shape [batch, num_attrs].
        z_tail: Tail community probabilities, shape [batch, num_attrs].

    Returns:
        PC2AuditResult with `log_prob` shaped [batch].
    """

    if z_head.shape != z_tail.shape:
        raise ValueError("z_head and z_tail must have the same shape")
    combined = 0.5 * (z_head + z_tail)
    combined = torch.clamp(combined, pc_model.smoothing_epsilon, 1.0 - pc_model.smoothing_epsilon)
    attr_probs = torch.stack([combined, 1.0 - combined], dim=-1)
    labels = torch.ones(z_head.size(0), device=z_head.device, dtype=torch.long)
    log_prob = pc_model.log_prob(attr_probs, labels)
    return PC2AuditResult(log_prob=log_prob)

"""Differentiable logic layer for DSLFM.

Applies soft logic constraints using configurable t-norms to encourage
consistency between symbolic evidence (rule bodies) and neural predictions
(rule heads). The layer emits differentiable penalties that can be composed
with KGE losses during joint training.

Design Pattern:
    - Strategy: t-norm selection (product, lukasiewicz, godel) governs how
      body literals are aggregated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class RuleDefinition:
    """Definition of a differentiable implication rule."""

    head_index: int
    body_indices: list[int] = field(default_factory=list)
    weight: float = 1.0


class DifferentiableRuleEncoder(nn.Module):
    """Encode fuzzy logic rules into differentiable penalties."""

    def __init__(
        self,
        t_norm: str = "product",
        smoothing: float = 1e-6,
        rules: Iterable[RuleDefinition] | None = None,
    ) -> None:
        """Create a logic encoder.

        Args:
            t_norm: T-norm used to aggregate body literals. Supported:
                product, lukasiewicz, godel.
            smoothing: Numerical floor applied to probabilities.
            rules: Iterable of rule definitions. If None, a default rule
                `domain_type_match ∧ range_type_match -> triple_plausibility`
                is used (indices 1,2 -> 0).
        """
        super().__init__()
        self.smoothing = float(smoothing)
        self.t_norm_name = t_norm.lower()
        self._t_norm = self._resolve_t_norm(self.t_norm_name)
        default_rules = [
            RuleDefinition(head_index=0, body_indices=[1, 2], weight=1.0),
        ]
        self.rules = list(rules) if rules is not None else default_rules

    def forward(self, attr_probs: torch.Tensor) -> torch.Tensor:
        """Compute per-sample logic penalties.

        Args:
            attr_probs: Tensor [batch, num_attrs, 2] with (p, 1-p) pairs.

        Returns:
            Tensor [batch] with average penalty across rules.
        """
        if attr_probs.dim() != 3 or attr_probs.size(2) != 2:
            raise ValueError(
                f"attr_probs must have shape [batch, num_attrs, 2], got {attr_probs.shape}"
            )

        probs = torch.clamp(attr_probs[..., 0], self.smoothing, 1.0 - self.smoothing)
        penalties = []
        for rule in self.rules:
            if not rule.body_indices:
                continue
            body_value = self._t_norm(probs[:, rule.body_indices])
            head_value = probs[:, rule.head_index]
            unmet_body = torch.relu(body_value - head_value)
            missing_support = head_value * torch.clamp(1.0 - body_value, min=0.0)
            penalty = (unmet_body + missing_support) * rule.weight
            penalties.append(penalty)

        if not penalties:
            return torch.zeros(
                attr_probs.size(0), device=attr_probs.device, dtype=attr_probs.dtype
            )

        stacked = torch.stack(penalties, dim=1)
        return torch.mean(stacked, dim=1)

    def _resolve_t_norm(self, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        if name == "product":
            return lambda x: torch.prod(x, dim=1)
        if name == "lukasiewicz":
            return lambda x: torch.clamp(
                torch.sum(x, dim=1) - (x.size(1) - 1), min=0.0, max=1.0
            )
        if name == "godel":
            return lambda x: torch.min(x, dim=1).values
        raise ValueError(f"Unsupported t_norm: {name}")

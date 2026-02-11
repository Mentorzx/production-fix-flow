"""KGE Strategy abstractions (Strategy/Factory patterns).

Provides strategy wrappers after consolidating on the DSLFM-KGC stack. Only the
DSLFM strategy is supported; legacy KGE paths were removed to avoid accidental
use outside deprecated modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


@dataclass
class KGEConfig:
    """Configuration wrapper for KGE strategies."""

    embedding_dim: int = 128
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return self.extra.get(key, default)


class KGEModelStrategy(ABC):
    """Abstract Strategy for Knowledge Graph Embedding models."""

    name: str = "base"

    def __init__(self, config: KGEConfig | None = None) -> None:
        self.config = config or KGEConfig()

    @abstractmethod
    def create_model(
        self,
        num_entities: int,
        num_relations: int,
        device: torch.device | str | None = None,
        relation_names: list[str] | None = None,
    ) -> nn.Module:
        """Create the model instance."""

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device if isinstance(device, torch.device) else torch.device(device)


class DSLFMStrategy(KGEModelStrategy):
    """Strategy implementation for DSLFM-KGC."""

    name = "dslfm-kgc"

    def __init__(self, config: KGEConfig | None = None) -> None:
        super().__init__(config)
        self.npc: nn.Module | None = None

    def create_model(
        self,
        num_entities: int,
        num_relations: int,
        device: torch.device | str | None = None,
        relation_names: list[str] | None = None,
    ) -> DSLFMKGCModel:
        dslfm_config = self._build_dslfm_config(
            num_entities, num_relations, relation_names
        )
        model = DSLFMKGCModel(dslfm_config, relation_names=relation_names)
        self.npc = model.pc_model
        return model.to(self._resolve_device(device))

    def compute_loss(
        self,
        model: nn.Module,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(model, DSLFMKGCModel):
            raise TypeError(
                f"DSLFMStrategy requires DSLFMKGCModel, got {type(model).__name__}"
            )
        heads = positive_triples[:, 0]
        relations = positive_triples[:, 1]
        tails = positive_triples[:, 2]
        result = model.compute_loss(
            heads=heads,
            relations=relations,
            tails=tails,
            use_inbatch_negatives=True,
        )
        loss = result.get("loss")
        if loss is None:
            raise RuntimeError("DSLFM compute_loss did not return a loss tensor")
        return loss

    def _build_dslfm_config(
        self,
        num_entities: int,
        num_relations: int,
        relation_names: list[str] | None,
    ) -> DSLFMKGCConfig:
        cfg = self.config
        extra = cfg.extra or {}
        return DSLFMKGCConfig(
            num_entities=num_entities,
            num_relations=num_relations,
            entity_dim=int(cfg.embedding_dim),
            feature_dim=int(cfg.embedding_dim),
            hidden_dim=int(extra.get("attr_hidden_dim", cfg.embedding_dim * 2)),
            ibp_alpha=float(extra.get("ibp_alpha", 1.0)),
            use_bert_relations=bool(
                extra.get("use_bert_relations", False and relation_names)
            ),
            bert_model=str(extra.get("bert_model", "bert-base-uncased")),
            temperature=float(extra.get("temperature", 0.5)),
            stochastic_latents=bool(extra.get("stochastic_latents", False)),
            encoder_dropout_p=float(extra.get("encoder_dropout_p", 0.0)),
            kl_weight=float(extra.get("kl_weight", 0.1)),
            sparsity_weight=float(extra.get("sparsity_weight", 0.01)),
            use_checkpointing=bool(extra.get("use_checkpointing", False)),
            sampler_type=str(extra.get("sampler_type", "self_adversarial")),
            sampler_temperature=float(extra.get("sampler_temperature", 1.0)),
            learnable_temperature=bool(extra.get("learnable_temperature", False)),
            contrastive_temperature=float(extra.get("contrastive_temperature", 0.07)),
            lambda_logic=float(extra.get("lambda_logic", 0.0)),
            t_norm=str(extra.get("t_norm", "product")),
            smoothing_epsilon=float(extra.get("smoothing_epsilon", 1e-6)),
            lambda_pc=float(extra.get("lambda_pc", 0.0)),
            pc_pruning_threshold=float(extra.get("pruning_threshold", 0.01)),
            pc_grow_noise=float(extra.get("grow_noise", 0.01)),
            pc_rebuild_every=int(extra.get("rebuild_every", 0)),
            pc_max_depth=(
                int(extra["max_circuit_depth"])
                if "max_circuit_depth" in extra
                else None
            ),
            triton_min_entities=int(extra.get("triton_min_entities", 1024)),
        )


__all__ = [
    "KGEConfig",
    "KGEModelStrategy",
    "DSLFMStrategy",
]

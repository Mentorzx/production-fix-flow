"""Model factory for KGC and PC2 models.

Centralizes model creation, weight initialization, and configuration
loading to ensure consistency across drivers and use cases.
"""

from __future__ import annotations

from typing import Any

import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel
from pff.domain.learning.pc.npc import NeuralProbabilisticCircuit
from pff.shared.core.logging import logger


class ModelFactory:
    """Unified factory for creating machine learning models in the PFF domain."""

    @staticmethod
    def create_kgc_model(
        config_dict: dict[str, Any],
        num_entities: int,
        num_relations: int,
        num_triples: int = 0,
        relation_names: list[str] | None = None,
        device: str | torch.device = "cpu",
    ) -> DSLFMKGCModel:
        """Create and initialize a DSLFM-KGC model."""
        config = DSLFMKGCConfig(
            num_entities=num_entities,
            num_relations=num_relations,
            num_triples=num_triples,
            entity_dim=config_dict.get("entity_dim", 256),
            feature_dim=config_dict.get("feature_dim", 256),
            max_communities=config_dict.get("max_communities", 128),
            hidden_dim=config_dict.get("hidden_dim", 512),
            ibp_alpha=config_dict.get("ibp_alpha", 1.0),
            use_bert_relations=config_dict.get("use_bert_relations", False),
            bert_model=config_dict.get("bert_model", "bert-base-uncased"),
            temperature=config_dict.get("temperature", 0.5),
            stochastic_latents=config_dict.get("stochastic_latents", False),
            kl_weight=config_dict.get("kl_weight", 0.1),
            free_bits=config_dict.get("free_bits", 0.125),
            sparsity_weight=config_dict.get("sparsity_weight", 0.01),
            sampler_type=config_dict.get("sampler_type", "degree_based"),
            contrastive_temperature=config_dict.get("contrastive_temperature", 0.07),
            lambda_logic=config_dict.get("lambda_logic", 0.0),
            lambda_pc=config_dict.get("lambda_pc", 0.0),
            nsc_cache_size=config_dict.get("nsc_cache_size", 64),
            nsc_sample_ratio=config_dict.get("nsc_sample_ratio", 0.5),
        )

        model = DSLFMKGCModel(config, relation_names=relation_names)
        model.to(device)

        logger.info(f"KGC Model created on {device}")
        return model

    @staticmethod
    def create_pc2_model(
        config_dict: dict[str, Any],
        num_attrs: int,
        device: str | torch.device = "cpu",
    ) -> NeuralProbabilisticCircuit:
        """Create and initialize a PC2 (Neural Probabilistic Circuit) model."""
        model = NeuralProbabilisticCircuit(
            num_attrs=num_attrs,
            smoothing_epsilon=config_dict.get("smoothing_epsilon", 1e-6),
            pruning_threshold=config_dict.get("pruning_threshold", 0.01),
            grow_noise=config_dict.get("grow_noise", 0.01),
            max_depth=config_dict.get("max_depth"),
            prune_every_n_steps=config_dict.get("prune_every_n_steps", 100),
        )
        model.to(device)

        logger.info(f"PC2 Model created on {device} with {num_attrs} attributes")
        return model

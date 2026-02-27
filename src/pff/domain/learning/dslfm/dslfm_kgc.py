"""DSLFM-KGC: Deep Sparse Latent Feature Model for Knowledge Graph Completion.

This module implements the full DSLFM-KGC architecture from the paper:
- Entity embeddings (learned, since we don't have text descriptions)
- Relation embeddings (BERT-based when available)
- VAE encoder with IBP prior for community discovery
- Stochastic Blockmodel decoder for triple scoring

Design Patterns:
    - Facade: DSLFMKGCModel provides unified interface to all components
    - Composite: Combines multiple sub-models
    - Strategy: Encoder/decoder can be swapped
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder_port import DecoderStrategy
from pff.shared.acceleration.triton_kernels import (
    TritonDotProductValidator,
    compute_ranks_from_scores_triton,
    is_triton_available,
)
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger
from pff.shared.ops.global_interrupt_manager import check_interruption

from .neg_sampling import (
    SamplerConfig,
    SamplerType,
    get_negative_sampler,
)
from .sbm_decoder import StochasticBlockmodelDecoder
from .vae import DSLFMVAEEncoder


class _BaseEvalBackend:
    def compute_ranks(
        self,
        scores: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """Execute compute ranks.



        Args:

            scores: Input value used by this callable.

            tails: Input value used by this callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        raise NotImplementedError


class _TorchEvalBackend(_BaseEvalBackend):
    def compute_ranks(
        self,
        scores: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """Execute compute ranks.



        Args:

            scores: Input value used by this callable.

            tails: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        true_scores = scores.gather(1, tails.unsqueeze(1))
        return (scores > true_scores).sum(dim=1) + 1


class _TritonEvalBackend(_BaseEvalBackend):
    def compute_ranks(
        self,
        scores: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ranks using a Triton streaming kernel."""
        return compute_ranks_from_scores_triton(scores, tails)


@dataclass
class DSLFMKGCConfig:
    """Configuration for DSLFM-KGC model."""

    num_entities: int
    num_relations: int
    num_triples: int = 0
    entity_dim: int = 256
    feature_dim: int = 256
    max_communities: int = 128
    hidden_dim: int = 512
    ibp_alpha: float = 1.0
    use_bert_relations: bool = False
    bert_model: str = "bert-base-uncased"
    temperature: float = 0.5
    stochastic_latents: bool = False
    encoder_dropout_p: float = 0.0
    kl_weight: float = 0.1
    free_bits: float = 0.125
    sparsity_weight: float = 0.01
    use_checkpointing: bool = False
    sampler_type: str = "degree_based"
    sampler_temperature: float = 1.0
    learnable_temperature: bool = False
    contrastive_temperature: float = 0.07
    lambda_logic: float = 0.0
    t_norm: str = "product"
    smoothing_epsilon: float = 1e-6
    lambda_pc: float = 0.0
    pc_pruning_threshold: float = 0.01
    pc_grow_noise: float = 0.01
    pc_rebuild_every: int = 0
    pc_max_depth: int | None = None
    pc_inbatch_rerank: bool = False
    pc_prune_every_n_steps: int = 100
    num_global_negatives: int = 0
    cache_global_negatives: bool = False
    global_negatives_refresh_steps: int = 50
    negative_sample_size: int = 0
    triton_min_entities: int = 1024
    logvar_clip_min: float = -20.0
    logvar_clip_max: float = 10.0
    nsc_cache_size: int = 64
    nsc_sample_ratio: float = 0.5
    community_weight: float = 1.0
    feature_weight: float = 0.0
    num_workers: int = 0
    triton_heuristic_high_mem: int = 24
    triton_heuristic_med_mem: int = 8
    triton_heuristic_batch_high: int = 2048
    triton_heuristic_batch_med: int = 512
    triton_heuristic_batch_low: int = 256


class DSLFMKGCModel(nn.Module):
    """Represent DSLFMKGCModel.



    Notes:

        Encapsulates behavior while preserving architecture boundaries.

    """

    def __init__(
        self,
        config: DSLFMKGCConfig,
        relation_names: list[str] | None = None,
    ) -> None:
        """Execute init.



        Args:

            config: Input value used by this callable.

            relation_names: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        super().__init__()
        self.config = config
        self.last_pc2_latency = 0.0
        self.eval_backend: _BaseEvalBackend = (
            _TritonEvalBackend() if is_triton_available() else _TorchEvalBackend()
        )

        self.entity_embedding = nn.Embedding(
            config.num_entities,
            config.entity_dim,
        )

        self.use_bert_relations = (
            config.use_bert_relations and relation_names is not None
        )

        if self.use_bert_relations:
            from .bert_encoder import TRANSFORMERS_AVAILABLE, RelationTextEncoder

            if TRANSFORMERS_AVAILABLE:
                try:
                    self.relation_encoder = RelationTextEncoder(
                        model_name=config.bert_model,
                        hidden_dim=config.entity_dim,
                        freeze_bert=True,
                    )
                    self.relation_names = relation_names
                    self._precomputed_relation_emb: torch.Tensor | None = None
                    logger.info(
                        f"Encoder BERT para relações ativado: {config.bert_model}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load BERT relation encoder: {e}")
                    self.use_bert_relations = False
            else:
                logger.warning(
                    "transformers not available, falling back to learned embeddings"
                )
                self.use_bert_relations = False

        if not self.use_bert_relations:
            self.relation_embedding = nn.Embedding(
                config.num_relations, config.entity_dim
            )

        self.vae_encoder = DSLFMVAEEncoder(
            input_dim=config.entity_dim,
            feature_dim=config.feature_dim,
            max_communities=config.max_communities,
            hidden_dim=config.hidden_dim,
            ibp_alpha=config.ibp_alpha,
            use_checkpointing=config.use_checkpointing,
            dropout_p=config.encoder_dropout_p,
            logvar_clip_min=config.logvar_clip_min,
            logvar_clip_max=config.logvar_clip_max,
        )
        if not config.stochastic_latents:
            self.vae_encoder.eval()

        self.logic_encoder = None
        if config.lambda_logic > 0:
            from .logic_layer import DifferentiableRuleEncoder

            self.logic_encoder = DifferentiableRuleEncoder(
                t_norm=config.t_norm,
                smoothing=config.smoothing_epsilon,
            )

        self.pc_model = None
        if config.lambda_pc > 0:
            from pff.domain.learning.pc.npc import NeuralProbabilisticCircuit

            self.pc_model = NeuralProbabilisticCircuit(
                num_attrs=config.max_communities,
                smoothing_epsilon=config.smoothing_epsilon,
                pruning_threshold=config.pc_pruning_threshold,
                grow_noise=config.pc_grow_noise,
                max_depth=config.pc_max_depth,
                prune_every_n_steps=config.pc_prune_every_n_steps,
            )

        self.decoder: DecoderStrategy = StochasticBlockmodelDecoder(
            num_communities=config.max_communities,
            feature_dim=config.feature_dim,
            num_relations=config.num_relations,
            community_weight=config.community_weight,
            feature_weight=config.feature_weight,
        )

        sampler_config = SamplerConfig(
            sampler_type=SamplerType(config.sampler_type),
            temperature=config.sampler_temperature,
            num_entities=config.num_entities,
            num_triples=config.num_triples,
            cache_size=config.nsc_cache_size,
            sample_ratio=config.nsc_sample_ratio,
        )
        self.negative_sampler = get_negative_sampler(
            sampler_type=config.sampler_type,
            config=sampler_config,
        )

        self._all_entity_features: torch.Tensor | None = None
        self._all_entity_communities: torch.Tensor | None = None
        self._entity_cache_version = 0
        self._faiss_index = None
        self._faiss_index_key: tuple[int, int] | None = None
        self._faiss_gpu_resources = None
        self._ann_backend = "faiss"
        self._cuvs_mode: str | None = None
        self._cuvs_modules: tuple[Any, ...] | None = None
        self._cuvs_search_params: Any | None = None
        self.base_model = SimpleNamespace(entity_embedding=self.entity_embedding)
        self._triton_min_entities = config.triton_min_entities

        if config.learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(math.log(config.contrastive_temperature))
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(math.log(config.contrastive_temperature)),
            )

        self._init_weights()

    def train(self, mode: bool = True) -> "DSLFMKGCModel":
        """Execute train.



        Args:

            mode: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        super().train(mode)
        if mode and not self.config.stochastic_latents:
            self.vae_encoder.eval()
        return self

    def maintenance(self) -> None:
        """Perform periodic maintenance on sub-models."""
        if self.pc_model is not None:
            self.pc_model.maintenance()

    def precompute_relation_embeddings(self, device: torch.device) -> None:
        """Execute precompute relation embeddings.



        Args:

            device: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if self.use_bert_relations and self._precomputed_relation_emb is None:
            if self.relation_names is None:
                logger.warning(
                    "relation_names is None, cannot precompute BERT embeddings"
                )
                return
            self._precomputed_relation_emb = (
                self.relation_encoder.precompute_relation_embeddings(
                    self.relation_names, device
                )
            )

    def get_relation_embeddings(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """Execute get relation embeddings.



        Args:

            relation_ids: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if self.use_bert_relations:
            if self._precomputed_relation_emb is None:
                self.precompute_relation_embeddings(relation_ids.device)
            if self._precomputed_relation_emb is not None:
                return self._precomputed_relation_emb[relation_ids]  # type: ignore[no-any-return]
        return self.relation_embedding(relation_ids)  # type: ignore[no-any-return]

    @property
    def effective_temperature(self) -> torch.Tensor:
        """Execute effective temperature.



        Returns:

            Return value produced by the callable.

        """

        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def _sample_global_negative_tail_ids(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor | None = None,
        tails: torch.Tensor | None = None,
        num_negatives: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sample global negative tail IDs (backward compatibility)."""
        num_entities = kwargs.get("num_entities", self.config.num_entities)
        if num_entities <= 1:
            raise ValueError("num_entities must be > 1 for negative sampling")

        if relations is None:
            relations = torch.zeros(
                heads.shape[0], dtype=torch.long, device=heads.device
            )
        if tails is None:
            tails = torch.zeros(heads.shape[0], dtype=torch.long, device=heads.device)

        neg_ids = self.negative_sampler.sample_negatives(
            heads, relations, tails, num_negatives=num_negatives
        )

        pos_tails = tails.view(-1, 1).expand(-1, num_negatives)
        for _ in range(num_negatives + 2):
            mask = neg_ids == pos_tails
            if not mask.any():
                break

            neg_ids = torch.where(mask, (neg_ids + 1) % int(num_entities), neg_ids)

        return neg_ids

    def _init_weights(self) -> None:
        """Execute init weights."""

        nn.init.xavier_uniform_(self.entity_embedding.weight)
        if not self.use_bert_relations:
            nn.init.xavier_uniform_(self.relation_embedding.weight)

    def encode_entities(
        self, entity_ids: torch.Tensor, temperature: float | None = None
    ) -> dict[str, torch.Tensor]:
        """Execute encode entities.



        Args:

            entity_ids: Input value used by this callable.

            temperature: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if temperature is None:
            temperature = self.config.temperature
        target_device = self.entity_embedding.weight.device
        if entity_ids.device != target_device:
            entity_ids = entity_ids.to(target_device, non_blocking=True)
        if entity_ids.dtype != torch.long:
            entity_ids = entity_ids.long()
        num_entities = int(self.entity_embedding.num_embeddings)
        if num_entities <= 0:
            raise ValueError("entity embedding table is empty")
        invalid_mask = (entity_ids < 0) | (entity_ids >= num_entities)
        if not torch._dynamo.is_compiling() and bool(invalid_mask.any().item()):
            min_id = int(entity_ids.min().item())
            max_id = int(entity_ids.max().item())
            logger.warning(
                f"Out-of-range entity IDs detected (min={min_id}, max={max_id}, limit={num_entities}). "
                "Applying modulo correction for CUDA-safe execution."
            )
        corrected_ids = torch.remainder(entity_ids, num_entities)
        entity_ids = torch.where(invalid_mask, corrected_ids, entity_ids)
        entity_emb = self.entity_embedding(entity_ids)
        return self.vae_encoder(entity_emb, temperature=temperature)  # type: ignore[no-any-return]

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        return_latents: bool = False,
        use_pc: bool = True,
    ) -> dict[str, Any]:
        """Execute forward.



        Args:

            heads: Input value used by this callable.

            relations: Input value used by this callable.

            tails: Input value used by this callable.

            return_latents: Optional input value.

            use_pc: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        effective_use_pc = bool(
            use_pc and self.pc_model is not None and self.config.lambda_pc > 0
        )
        head_latents = self.encode_entities(heads)
        tail_latents = self.encode_entities(tails)
        decoder_scores = self.decoder.forward(
            z_head=head_latents["communities"],
            z_tail=tail_latents["communities"],
            f_head=head_latents["features"],
            f_tail=tail_latents["features"],
            relations=relations,
        )
        scores = decoder_scores
        if effective_use_pc:
            pc_log = self._pc_log_prob_pairwise(
                head_latents["communities"], tail_latents["communities"]
            )
            if pc_log is not None:
                scores = scores + self.config.lambda_pc * pc_log
        attr_probs = torch.stack(
            [head_latents["communities"], 1.0 - head_latents["communities"]], dim=-1
        )
        result: dict[str, Any] = {
            "scores": scores,
            "decoder_scores": decoder_scores,
            "attr_probs": attr_probs,
        }
        if return_latents:
            result["head_latents"] = head_latents
            result["tail_latents"] = tail_latents
        return result

    def _map_unique_entities(
        self,
        heads: torch.Tensor,
        tails: torch.Tensor,
        temperature: float | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Execute map unique entities.



        Args:

            heads: Input value used by this callable.

            tails: Input value used by this callable.

            temperature: Optional input value.



        Returns:

            Return value produced by the callable.

        """

        all_ents = torch.cat([heads, tails])
        unique_ents, inverse_indices = torch.unique(all_ents, return_inverse=True)

        unique_latents = self.encode_entities(unique_ents, temperature=temperature)
        z_unique = unique_latents["communities"]
        f_unique = unique_latents["features"]

        n_triples = heads.shape[0]
        head_indices = inverse_indices[:n_triples].to(
            z_unique.device, non_blocking=True
        )
        tail_indices = inverse_indices[n_triples:].to(
            z_unique.device, non_blocking=True
        )

        head_latents = {
            "communities": z_unique[head_indices],
            "features": f_unique[head_indices],
        }
        tail_latents = {
            "communities": z_unique[tail_indices],
            "features": f_unique[tail_indices],
        }

        if "mu" in unique_latents:
            head_latents["mu"] = unique_latents["mu"][head_indices]
            head_latents["logvar"] = unique_latents["logvar"][head_indices]
            tail_latents["mu"] = unique_latents["mu"][tail_indices]
            tail_latents["logvar"] = unique_latents["logvar"][tail_indices]

        return head_latents, tail_latents

    def score_triples_batch(self, triples: torch.Tensor) -> torch.Tensor:
        """Score a batch of triples (h, r, t) with high-performance optimizations."""
        if triples.numel() == 0:
            return torch.empty((0,), device=triples.device)

        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]

        head_latents, tail_latents = self._map_unique_entities(heads, tails)

        z_heads = head_latents["communities"]
        z_tails = tail_latents["communities"]
        f_heads = head_latents["features"]
        f_tails = tail_latents["features"]

        scores = self.decoder.forward(
            z_head=z_heads,
            z_tail=z_tails,
            f_head=f_heads,
            f_tail=f_tails,
            relations=relations,
        )

        if self.pc_model is not None and self.config.lambda_pc > 0:
            pc_log = self._pc_log_prob_pairwise(z_heads, z_tails)
            if pc_log is not None:
                scores = scores + self.config.lambda_pc * pc_log

        return scores

    def compute_loss(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        use_inbatch_negatives: bool = True,
        temperature: float | torch.Tensor | None = None,
        entity_temperature: float | None = None,
        regularization_scale: float = 1.0,
        known_positive_mask: torch.Tensor | None = None,
        triple_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Execute compute loss.



        Args:

            heads: Input value used by this callable.

            relations: Input value used by this callable.

            tails: Input value used by this callable.

            use_inbatch_negatives: Optional input value.

            temperature: Optional input value.

            entity_temperature: Optional input value.

            regularization_scale: Optional input value.

            known_positive_mask: Optional input value.

            triple_indices: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        batch_size = heads.shape[0]
        target_device = self.entity_embedding.weight.device
        if heads.device != target_device:
            heads = heads.to(target_device, non_blocking=True)
        if relations.device != target_device:
            relations = relations.to(target_device, non_blocking=True)
        if tails.device != target_device:
            tails = tails.to(target_device, non_blocking=True)
        if heads.dtype != torch.long:
            heads = heads.long()
        if relations.dtype != torch.long:
            relations = relations.long()
        if tails.dtype != torch.long:
            tails = tails.long()
        num_entities = int(self.entity_embedding.num_embeddings)
        num_relations = int(
            self.relation_embedding.num_embeddings
            if not self.use_bert_relations
            else self.config.num_relations
        )
        if num_entities <= 0:
            raise ValueError("num_entities must be > 0")
        if num_relations <= 0:
            raise ValueError("num_relations must be > 0")
        invalid_entities = (
            (heads < 0)
            | (heads >= num_entities)
            | (tails < 0)
            | (tails >= num_entities)
        )
        if invalid_entities.any():
            min_h = int(heads.min().item())
            max_h = int(heads.max().item())
            min_t = int(tails.min().item())
            max_t = int(tails.max().item())
            logger.warning(
                "Out-of-range triple entity IDs detected before loss computation "
                f"(heads=[{min_h}, {max_h}] tails=[{min_t}, {max_t}] limit={num_entities}). "
                "Applying modulo correction."
            )
            heads = torch.remainder(heads, num_entities)
            tails = torch.remainder(tails, num_entities)
        invalid_relations = (relations < 0) | (relations >= num_relations)
        if invalid_relations.any():
            min_r = int(relations.min().item())
            max_r = int(relations.max().item())
            logger.warning(
                f"Out-of-range relation IDs detected before loss computation "
                f"(relations=[{min_r}, {max_r}] limit={num_relations}). Applying modulo correction."
            )
            relations = torch.remainder(relations, num_relations)
        if (
            known_positive_mask is not None
            and known_positive_mask.device != target_device
        ):
            known_positive_mask = known_positive_mask.to(
                target_device, non_blocking=True
            )
        if triple_indices is not None and triple_indices.device != target_device:
            triple_indices = triple_indices.to(target_device, non_blocking=True)
        if temperature is None:
            temperature = self.effective_temperature

        head_latents, tail_latents = self._map_unique_entities(
            heads, tails, temperature=entity_temperature
        )

        community_probs = torch.cat(
            [head_latents["communities"], tail_latents["communities"]], dim=0
        )
        attr_probs = torch.stack([community_probs, 1.0 - community_probs], dim=-1)

        if use_inbatch_negatives:
            all_scores = self._score_all_pairs(
                head_latents,
                tail_latents,
                relations,
                use_pc=self.config.pc_inbatch_rerank,
            )
            pos_scores, neg_scores, neg_ids = (
                self.negative_sampler.get_positive_negative_scores(
                    all_scores, tails, known_positive_mask=known_positive_mask
                )
            )

            global_neg_k = self.config.num_global_negatives
            if global_neg_k > 0:
                neg_tail_ids_global = self.negative_sampler.sample_negatives(
                    heads,
                    relations,
                    tails,
                    num_negatives=global_neg_k,
                    triple_indices=triple_indices,
                )

                neg_tail_latents = self.encode_entities(
                    neg_tail_ids_global.reshape(-1), temperature=entity_temperature
                )
                z_tail_neg = neg_tail_latents["communities"].view(
                    batch_size, global_neg_k, -1
                )
                f_tail_neg = neg_tail_latents["features"].view(
                    batch_size, global_neg_k, -1
                )

                z_head = head_latents["communities"]
                f_head = head_latents["features"]

                z_head_rep = (
                    z_head.unsqueeze(1)
                    .expand(-1, global_neg_k, -1)
                    .reshape(-1, z_head.shape[-1])
                )
                f_head_rep = (
                    f_head.unsqueeze(1)
                    .expand(-1, global_neg_k, -1)
                    .reshape(-1, f_head.shape[-1])
                )
                relations_rep = (
                    relations.unsqueeze(1).expand(-1, global_neg_k).reshape(-1)
                )

                neg_scores_global = self.decoder.forward(
                    z_head=z_head_rep,
                    z_tail=z_tail_neg.reshape(-1, z_tail_neg.shape[-1]),
                    f_head=f_head_rep,
                    f_tail=f_tail_neg.reshape(-1, f_tail_neg.shape[-1]),
                    relations=relations_rep,
                ).view(batch_size, global_neg_k)

                neg_scores = torch.cat([neg_scores, neg_scores_global], dim=1)
                neg_ids = torch.cat([neg_ids, neg_tail_ids_global], dim=1)

            neg_cap = self.config.negative_sample_size
            if neg_cap > 0 and neg_scores.shape[1] > neg_cap:
                from pff.shared.acceleration.triton_kernels import (
                    fused_random_subsample_triton,
                )

                neg_scores = fused_random_subsample_triton(neg_scores, k=neg_cap)

            self.negative_sampler.update_cache(
                heads,
                relations,
                tails,
                neg_ids,
                neg_scores,
                triple_indices=triple_indices,
            )
            contrastive_loss = self.negative_sampler.contrastive_loss(
                pos_scores, neg_scores, temperature
            )
        else:
            pos_scores = self.decoder.forward(
                z_head=head_latents["communities"],
                z_tail=tail_latents["communities"],
                f_head=head_latents["features"],
                f_tail=tail_latents["features"],
                relations=relations,
            )
            contrastive_loss = -F.logsigmoid(pos_scores).mean()

        kl_losses = self.vae_encoder.kl_loss(
            mu=torch.cat([head_latents["mu"], tail_latents["mu"]], dim=0),
            logvar=torch.cat([head_latents["logvar"], tail_latents["logvar"]], dim=0),
            community_probs=torch.cat(
                [head_latents["communities"], tail_latents["communities"]], dim=0
            ),
            free_bits=self.config.free_bits,
        )

        sparsity_loss = (
            head_latents["communities"].abs().mean()
            + tail_latents["communities"].abs().mean()
        ) / 2

        logic_penalty = (
            self.logic_encoder(attr_probs).mean() if self.logic_encoder else None
        )

        pc_penalty = None
        if self.pc_model:
            pc_penalty = self.pc_model(
                attr_probs,
                torch.ones(
                    attr_probs.size(0), device=attr_probs.device, dtype=torch.long
                ),
            ).mean()
            self._last_pc_latency = 0.0
        else:
            self._last_pc_latency = 0.0

        total_loss = (
            contrastive_loss
            + self.config.kl_weight * kl_losses["kl_loss"]
            + self.config.sparsity_weight * sparsity_loss
        )
        reg_scale = max(0.0, min(1.0, regularization_scale))
        if logic_penalty is not None:
            total_loss += reg_scale * self.config.lambda_logic * logic_penalty
        if pc_penalty is not None:
            total_loss += reg_scale * self.config.lambda_pc * pc_penalty

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "kl_gaussian": kl_losses["kl_gaussian"],
            "kl_ibp": kl_losses["kl_ibp"],
            "sparsity_loss": sparsity_loss,
            "pc_penalty": (
                pc_penalty
                if pc_penalty is not None
                else torch.tensor(0.0, device=total_loss.device)
            ),
        }

    def _score_all_pairs(self, head_latents, tail_latents, relations, use_pc=True):
        """Execute score all pairs.



        Args:

            head_latents: Input value used by this callable.

            tail_latents: Input value used by this callable.

            relations: Input value used by this callable.

            use_pc: Optional input value.



        Returns:

            Return value produced by the callable.

        """

        z_heads, f_heads = head_latents["communities"], head_latents["features"]
        z_tails, f_tails = tail_latents["communities"], tail_latents["features"]
        all_scores = self.decoder.score_all_tails(
            z_head=z_heads,
            f_head=f_heads,
            relations=relations,
            all_z=z_tails,
            all_f=f_tails,
        )
        if use_pc and self.pc_model and self.config.lambda_pc > 0:
            pc_log = self._pc_log_prob_matrix(z_heads, z_tails)
            if pc_log is not None:
                all_scores += self.config.lambda_pc * pc_log
        return all_scores

    def _pc_log_prob_matrix(self, z_heads, all_z, chunk_size=1024):
        """Execute pc log prob matrix.



        Args:

            z_heads: Input value used by this callable.

            all_z: Input value used by this callable.

            chunk_size: Optional input value.



        Returns:

            Return value produced by the callable.

        """

        if not self.pc_model:
            return None
        import time

        timer_enabled = not torch._dynamo.is_compiling()
        start_t = time.perf_counter() if timer_enabled else 0.0

        eps = self.config.smoothing_epsilon
        num_heads = z_heads.shape[0]
        num_tails = all_z.shape[0]
        device = z_heads.device

        if device.type == "cuda":
            try:
                from pff.shared.acceleration.triton_kernels import (
                    pc2_matrix_forward_triton,
                )

                max_elements = 64 * 1024 * 1024
                heads_per_chunk = max(1, max_elements // num_tails)

                results = []
                for i in range(0, num_heads, heads_per_chunk):
                    z_h_chunk = z_heads[i : i + heads_per_chunk]
                    chunk_res = pc2_matrix_forward_triton(
                        z_h_chunk,
                        all_z,
                        torch.tensor(
                            self.pc_model.parents,
                            device=device,
                            dtype=torch.long,
                        ),
                        self.pc_model.root_probs,
                        self.pc_model.cond_probs,
                        torch.log_softmax(self.pc_model.label_logits, dim=0),
                    )
                    results.append(chunk_res)

                res_cat = torch.cat(results, dim=0)
                if timer_enabled:
                    self.last_pc2_latency = (time.perf_counter() - start_t) * 1000
                return res_cat
            except Exception:
                pass

        max_tails_chunk = 5000 if device.type == "cpu" else 1000
        max_heads_chunk = 128 if device.type == "cpu" else 64

        head_results = []
        for i in range(0, num_heads, max_heads_chunk):
            end_h = min(i + max_heads_chunk, num_heads)
            z_h_chunk = z_heads[i:end_h]

            tail_results = []
            for j in range(0, num_tails, max_tails_chunk):
                end_t = min(j + max_tails_chunk, num_tails)
                z_t_chunk = all_z[j:end_t]

                combined = torch.clamp(
                    0.5 * (z_h_chunk.unsqueeze(1) + z_t_chunk.unsqueeze(0)),
                    eps,
                    1.0 - eps,
                )
                attr_probs = torch.stack([combined, 1.0 - combined], dim=-1)
                labels = torch.ones(combined.shape[:2], device=device, dtype=torch.long)

                res = self.pc_model.log_prob(attr_probs, labels)
                tail_results.append(res)

            head_results.append(torch.cat(tail_results, dim=1))

        final_res = torch.cat(head_results, dim=0)
        if timer_enabled:
            self.last_pc2_latency = (time.perf_counter() - start_t) * 1000
        return final_res

    def _pc_log_prob_pairwise(self, z_heads, z_tails):
        """Execute pc log prob pairwise.



        Args:

            z_heads: Input value used by this callable.

            z_tails: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not self.pc_model:
            return None
        import time

        timer_enabled = not torch._dynamo.is_compiling()
        start_t = time.perf_counter() if timer_enabled else 0.0

        eps = self.config.smoothing_epsilon
        combined = torch.clamp(0.5 * (z_heads + z_tails), eps, 1.0 - eps)
        attr_probs = torch.stack([combined, 1.0 - combined], dim=-1)
        labels = torch.ones(z_heads.shape[0], device=z_heads.device, dtype=torch.long)
        res = self.pc_model.log_prob(attr_probs, labels)

        if timer_enabled:
            self.last_pc2_latency = (time.perf_counter() - start_t) * 1000
        return res

    @torch.no_grad()
    def evaluate(
        self,
        eval_triples: torch.Tensor,
        batch_size: int = 512,
        filter_fn: (
            Callable[
                [
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    bool,
                ],
                torch.Tensor,
            ]
            | None
        ) = None,
        rerank_top_k: int | None = None,
        use_faiss_eval: bool = False,
        faiss_candidate_k: int = 1024,
        score_all_tails_chunk_size: int = 20_000,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Full link prediction evaluation (MRR, Hits@k)."""
        self.eval()
        device = self.entity_embedding.weight.device
        all_z, all_f = self._resolve_eval_latents(
            batch_size=batch_size,
            refresh_cache=bool(kwargs.get("refresh_cache", False)),
        )

        if use_faiss_eval:
            from pff.domain.learning.ml.ann_evaluator import ANNConfig, should_use_ann

            ann_cfg = ANNConfig.from_defaults()
            if not should_use_ann(int(self.config.num_entities), ann_cfg):
                logger.info(
                    "Desativando avaliacao ANN/FAISS para grafo pequeno "
                    f"(entidades={self.config.num_entities}, limiar={ann_cfg.threshold_entities})"
                )
                use_faiss_eval = False

        if use_faiss_eval:
            self._ensure_faiss_index(all_f)

        if eval_triples.numel() == 0:
            logger.warning(
                "Validation skipped: eval_triples is empty; returning neutral ranking metrics."
            )
            return {
                "mrr": 0.0,
                "hits@1": 0.0,
                "hits@3": 0.0,
                "hits@10": 0.0,
                "ap@10": 0.0,
            }

        ranks: list[torch.Tensor] = []
        for i in range(0, len(eval_triples), batch_size):
            check_interruption()
            batch = eval_triples[i : i + batch_size].to(device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            head_latents = self.encode_entities(h)
            z_h, f_h = head_latents["communities"], head_latents["features"]

            true_scores = self._compute_true_scores(
                z_h=z_h,
                f_h=f_h,
                relations=r,
                tails=t,
                all_z=all_z,
                all_f=all_f,
            )

            batch_ranks = torch.ones(len(h), device=device, dtype=torch.int32)

            if use_faiss_eval:
                ranks.append(
                    self._evaluate_batch_with_faiss(
                        base_ranks=batch_ranks,
                        heads=h,
                        z_h=z_h,
                        f_h=f_h,
                        relations=r,
                        tails=t,
                        true_scores=true_scores,
                        filter_fn=filter_fn,
                        faiss_candidate_k=faiss_candidate_k,
                        rerank_top_k=rerank_top_k,
                    )
                )
                continue

            if self._can_use_triton_for_eval(device):
                ranks.append(
                    self._evaluate_batch_with_triton(
                        heads=h,
                        z_h=z_h,
                        f_h=f_h,
                        relations=r,
                        tails=t,
                        true_scores=true_scores,
                        all_z=all_z,
                        all_f=all_f,
                        filter_fn=filter_fn,
                    )
                )
                continue

            ranks.append(
                self._evaluate_batch_full_scan(
                    base_ranks=batch_ranks,
                    heads=h,
                    z_h=z_h,
                    f_h=f_h,
                    relations=r,
                    tails=t,
                    true_scores=true_scores,
                    all_z=all_z,
                    all_f=all_f,
                    filter_fn=filter_fn,
                    score_all_tails_chunk_size=score_all_tails_chunk_size,
                )
            )

        if hasattr(self, "_triton_validator_cache"):
            del self._triton_validator_cache

        return self._compute_ranking_metrics(torch.cat(ranks).float())

    def _resolve_eval_latents(
        self, *, batch_size: int, refresh_cache: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute resolve eval latents.



        Args:

            batch_size: Input value used by this callable.

            refresh_cache: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if self._all_entity_features is None or refresh_cache:
            self.precompute_entity_latents(batch_size=batch_size)
        all_z = self._all_entity_communities
        all_f = self._all_entity_features
        assert all_z is not None, "Entity communities must be precomputed"
        assert all_f is not None, "Entity features must be precomputed"
        return all_z, all_f

    def _compute_true_scores(
        self,
        *,
        z_h: torch.Tensor,
        f_h: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        all_z: torch.Tensor,
        all_f: torch.Tensor,
    ) -> torch.Tensor:
        """Execute compute true scores.



        Args:

            z_h: Input value used by this callable.

            f_h: Input value used by this callable.

            relations: Input value used by this callable.

            tails: Input value used by this callable.

            all_z: Input value used by this callable.

            all_f: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        true_scores = self.decoder.forward(
            z_head=z_h,
            z_tail=all_z[tails],
            f_head=f_h,
            f_tail=all_f[tails],
            relations=relations,
        )
        if self.pc_model and self.config.lambda_pc > 0:
            pc_log_true = self._pc_log_prob_pairwise(z_h, all_z[tails])
            if pc_log_true is not None:
                true_scores = true_scores + self.config.lambda_pc * pc_log_true
        return true_scores

    def _evaluate_batch_with_faiss(
        self,
        *,
        base_ranks: torch.Tensor,
        heads: torch.Tensor,
        z_h: torch.Tensor,
        f_h: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        true_scores: torch.Tensor,
        filter_fn: (
            Callable[
                [
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    bool,
                ],
                torch.Tensor,
            ]
            | None
        ),
        faiss_candidate_k: int,
        rerank_top_k: int | None,
    ) -> torch.Tensor:
        """Execute evaluate batch with faiss.



        Args:

            base_ranks: Input value used by this callable.

            heads: Input value used by this callable.

            z_h: Input value used by this callable.

            f_h: Input value used by this callable.

            relations: Input value used by this callable.

            tails: Input value used by this callable.

            true_scores: Input value used by this callable.

            filter_fn: Input value used by this callable.

            faiss_candidate_k: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        cand_scores, cand_idx = self._score_faiss_candidates(
            z_h, f_h, relations, faiss_candidate_k
        )
        if filter_fn:
            cand_scores = filter_fn(
                cand_scores, heads, relations, cand_idx, tails, False
            )
        if rerank_top_k is not None and 0 < rerank_top_k < cand_scores.shape[1]:
            top_scores, top_idx = torch.topk(cand_scores, k=rerank_top_k, dim=1)
            cand_idx = cand_idx.gather(1, top_idx)
            cand_scores = top_scores
        tails_exp = tails.unsqueeze(1)
        has_true = (cand_idx == tails_exp).any(dim=1)
        better_in_cand = (cand_scores > true_scores.unsqueeze(1)).sum(dim=1)
        approx_rank = torch.where(
            has_true,
            better_in_cand + 1,
            torch.full_like(better_in_cand, cand_scores.shape[1] + 1),
        )
        return base_ranks - 1 + approx_rank.to(torch.int32)

    def _can_use_triton_for_eval(self, device: torch.device) -> bool:
        return (
            is_triton_available()
            and device.type == "cuda"
            and isinstance(self.decoder, StochasticBlockmodelDecoder)
            and (not self.pc_model or self.config.lambda_pc <= 0)
        )

    def _ensure_triton_validator(
        self, *, all_z: torch.Tensor, all_f: torch.Tensor, device: torch.device
    ) -> None:
        """Execute ensure triton validator.



        Args:

            all_z: Input value used by this callable.

            all_f: Input value used by this callable.

            device: Input value used by this callable.

        """

        if hasattr(self, "_triton_validator_cache"):
            return
        sbm_decoder: Any = self.decoder
        w_c = torch.sqrt(torch.abs(sbm_decoder.community_weight))
        w_f = torch.sqrt(torch.abs(sbm_decoder.feature_weight))
        e_c = all_z * w_c
        e_f = F.normalize(all_f, p=2, dim=-1) * w_f
        e_b = torch.ones(self.config.num_entities, 1, device=device, dtype=all_z.dtype)
        entities_proj = torch.cat([e_c, e_f, e_b], dim=1)
        self._triton_validator_cache = TritonDotProductValidator(
            entities_proj, device=str(device)
        )

    def _evaluate_batch_with_triton(
        self,
        *,
        heads: torch.Tensor,
        z_h: torch.Tensor,
        f_h: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        true_scores: torch.Tensor,
        all_z: torch.Tensor,
        all_f: torch.Tensor,
        filter_fn: (
            Callable[
                [
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    bool,
                ],
                torch.Tensor,
            ]
            | None
        ),
    ) -> torch.Tensor:
        """Execute evaluate batch with triton.



        Args:

            heads: Input value used by this callable.

            z_h: Input value used by this callable.

            f_h: Input value used by this callable.

            relations: Input value used by this callable.

            tails: Input value used by this callable.

            true_scores: Input value used by this callable.

            all_z: Input value used by this callable.

            all_f: Input value used by this callable.

            filter_fn: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        device = tails.device
        self._ensure_triton_validator(all_z=all_z, all_f=all_f, device=device)
        sbm_dec: Any = self.decoder
        w_c = torch.sqrt(torch.abs(sbm_dec.community_weight))
        w_f = torch.sqrt(torch.abs(sbm_dec.feature_weight))
        w_r = sbm_dec.W[relations]
        q_c = torch.bmm(z_h.unsqueeze(1), w_r).squeeze(1) * w_c
        f_h_norm = F.normalize(f_h, p=2, dim=-1)
        q_f = (
            torch.mm(f_h_norm, sbm_dec.feature_bilinear.weight.squeeze(0))
            if sbm_dec.use_bilinear
            else f_h_norm
        ) * w_f
        q_b = sbm_dec.relation_bias[relations].unsqueeze(1)
        queries_proj = torch.cat([q_c, q_f, q_b], dim=1)
        batch_ranks = self._triton_validator_cache.compute_ranks(queries_proj, tails)
        if filter_fn is None:
            return batch_ranks
        correction = filter_fn(
            true_scores,
            heads,
            relations,
            torch.empty(0, device=device),
            tails,
            True,
        )
        corrected_ranks = batch_ranks - correction
        return torch.clamp(corrected_ranks, min=1).to(torch.int32)

    def _evaluate_batch_full_scan(
        self,
        *,
        base_ranks: torch.Tensor,
        heads: torch.Tensor,
        z_h: torch.Tensor,
        f_h: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        true_scores: torch.Tensor,
        all_z: torch.Tensor,
        all_f: torch.Tensor,
        filter_fn: (
            Callable[
                [
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    bool,
                ],
                torch.Tensor,
            ]
            | None
        ),
        score_all_tails_chunk_size: int,
    ) -> torch.Tensor:
        """Execute evaluate batch full scan.



        Args:

            base_ranks: Input value used by this callable.

            heads: Input value used by this callable.

            z_h: Input value used by this callable.

            f_h: Input value used by this callable.

            relations: Input value used by this callable.

            tails: Input value used by this callable.

            true_scores: Input value used by this callable.

            all_z: Input value used by this callable.

            all_f: Input value used by this callable.

            filter_fn: Input value used by this callable.

            score_all_tails_chunk_size: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        device = tails.device
        for start in range(0, self.config.num_entities, score_all_tails_chunk_size):
            end = min(start + score_all_tails_chunk_size, self.config.num_entities)
            chunk_scores = self.decoder.score_all_tails(
                z_head=z_h,
                f_head=f_h,
                relations=relations,
                all_z=all_z[start:end],
                all_f=all_f[start:end],
            )
            if self.pc_model and self.config.lambda_pc > 0:
                pc_log_chunk = self._pc_log_prob_matrix(z_h, all_z[start:end])
                if pc_log_chunk is not None:
                    chunk_scores = chunk_scores + self.config.lambda_pc * pc_log_chunk
            if filter_fn is not None:
                candidates = torch.arange(start, end, device=device)
                chunk_scores = filter_fn(
                    chunk_scores,
                    heads,
                    relations,
                    candidates,
                    tails,
                    False,
                )
            base_ranks += (
                (chunk_scores > true_scores.unsqueeze(1)).sum(dim=1).to(torch.int32)
            )
        return base_ranks

    @staticmethod
    def _compute_ranking_metrics(all_ranks: torch.Tensor) -> dict[str, float]:
        """Execute compute ranking metrics.



        Args:

            all_ranks: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        ranks = all_ranks.float()
        ranks = torch.where(torch.isfinite(ranks), ranks, torch.ones_like(ranks))
        ranks = torch.clamp(ranks, min=1.0)
        mrr = (1.0 / ranks).mean().item()
        hits_at_1 = (ranks <= 1).float().mean().item()
        hits_at_3 = (ranks <= 3).float().mean().item()
        hits_at_10 = (ranks <= 10).float().mean().item()
        ap_at_10 = (1.0 / ranks.clamp(max=10)).mean().item()
        return {
            "mrr": mrr,
            "hits@1": hits_at_1,
            "hits@3": hits_at_3,
            "hits@10": hits_at_10,
            "ap@10": ap_at_10,
        }

    def _ensure_faiss_index(self, features: torch.Tensor) -> None:
        """Execute ensure faiss index.



        Args:

            features: Input value used by this callable.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

        from pff.domain.learning.ml.ann_evaluator import (
            ANNConfig,
            _prepare_cuvs_runtime,
            ann_backend_available,
            build_faiss_index,
        )

        cfg = ANNConfig.from_defaults()
        backend = str(getattr(cfg, "backend", "faiss")).lower().strip()
        if self._faiss_index is not None and self._faiss_index_key == features.shape:
            if backend == self._ann_backend:
                return
        if not ann_backend_available(backend):
            raise ImportError(
                f"ANN backend '{backend}' not installed. "
                "Instale dependencias do backend configurado em config/models/dslfm.yaml."
            )
        self._ann_backend = backend
        self._faiss_cfg = cfg
        feat_np = features.detach().cpu().numpy().astype("float32")
        metric = str(getattr(cfg, "metric", "ip")).lower()
        self._faiss_metric = metric
        normalize = metric == "ip"
        self._faiss_normalize = normalize
        if normalize:
            feat_norms = np.linalg.norm(feat_np, axis=1, keepdims=True)
            np.divide(feat_np, np.maximum(feat_norms, 1e-12), out=feat_np)
        dim = feat_np.shape[1]
        num_entities = feat_np.shape[0]

        if backend == "faiss":
            self._faiss_index, _ = build_faiss_index(feat_np, cfg, metric=metric)
            self._cuvs_mode = None
            self._cuvs_modules = None
            self._cuvs_search_params = None
        elif backend == "scann":
            import scann  # type: ignore

            index_type = cfg.index_type.lower()
            distance = "dot_product" if metric == "ip" else "squared_l2"
            num_leaves_cfg = int(getattr(cfg, "scann_num_leaves", 0))
            leaves_search_cfg = int(getattr(cfg, "scann_num_leaves_to_search", 0))
            num_leaves = (
                max(1, num_leaves_cfg)
                if num_leaves_cfg > 0
                else max(1, min(cfg.nlist, max(1, int(np.sqrt(max(2, num_entities))))))
            )
            leaves_to_search = (
                max(1, min(leaves_search_cfg, num_leaves))
                if leaves_search_cfg > 0
                else max(1, min(cfg.nprobe, num_leaves))
            )
            candidate_count = max(128, int(cfg.nprobe) * 8)
            reorder_k = int(getattr(cfg, "scann_reorder_k", 0))
            scann_builder: Any = scann.scann_ops_pybind.builder(
                feat_np, candidate_count, distance
            )
            if index_type == "flat":
                self._faiss_index = scann_builder.score_brute_force().build()
            elif index_type in {"ivf", "ivfpq", "hnsw"}:
                scorer = scann_builder.tree(
                    num_leaves=num_leaves,
                    num_leaves_to_search=leaves_to_search,
                    training_sample_size=min(num_entities, 250000),
                ).score_ah(2, anisotropic_quantization_threshold=0.2)
                if reorder_k > 0:
                    scorer = scorer.reorder(reorder_k)
                else:
                    scorer = scorer.reorder(max(32, cfg.ef_search))
                self._faiss_index = scorer.build()
            else:
                raise ValueError(
                    f"Unsupported ANN index_type for ScaNN: {cfg.index_type}"
                )
            self._cuvs_mode = None
            self._cuvs_modules = None
            self._cuvs_search_params = None
        elif backend == "cuvs":
            _prepare_cuvs_runtime()
            import cupy as cp  # type: ignore
            from cuvs.neighbors import brute_force, cagra, ivf_flat, ivf_pq  # type: ignore

            metric_name = "inner_product" if metric == "ip" else "sqeuclidean"
            dataset_cp = cp.asarray(feat_np)
            index_type = cfg.index_type.lower()
            if index_type in {"flat", "hnsw"}:
                self._faiss_index = brute_force.build(dataset_cp, metric=metric_name)
                self._cuvs_mode = "brute_force"
                self._cuvs_search_params = None
            elif index_type == "ivf":
                # FAISS/cuVS recommends at least 39 * nlist training points for IVF
                max_nlist = max(1, num_entities // 39)
                n_lists = max(1, min(cfg.nlist, max_nlist))
                if n_lists < cfg.nlist:
                    logger.debug(
                        f"Reducing IVF nlist from {cfg.nlist} to {n_lists} (num_entities={num_entities})"
                    )
                index_params = ivf_flat.IndexParams(n_lists=n_lists, metric=metric_name)
                search_params = ivf_flat.SearchParams(n_probes=min(cfg.nprobe, n_lists))
                self._faiss_index = ivf_flat.build(index_params, dataset_cp)
                self._cuvs_mode = "ivf_flat"
                self._cuvs_search_params = search_params
            elif index_type == "ivfpq":
                # FAISS/cuVS recommends at least 39 * nlist training points for IVF
                max_nlist = max(1, num_entities // 39)
                n_lists = max(1, min(cfg.nlist, max_nlist))
                if n_lists < cfg.nlist:
                    logger.debug(
                        f"Reducing IVF-PQ nlist from {cfg.nlist} to {n_lists} (num_entities={num_entities})"
                    )
                try:
                    index_params = ivf_pq.IndexParams(
                        n_lists=n_lists,
                        metric=metric_name,
                        m=cfg.M,
                        nbits=cfg.pq_bits,
                    )
                except TypeError:
                    index_params = ivf_pq.IndexParams(
                        n_lists=n_lists,
                        metric=metric_name,
                        pq_bits=cfg.pq_bits,
                    )
                search_params = ivf_pq.SearchParams(n_probes=min(cfg.nprobe, n_lists))
                self._faiss_index = ivf_pq.build(index_params, dataset_cp)
                self._cuvs_mode = "ivf_pq"
                self._cuvs_search_params = search_params
            elif index_type == "cagra":
                graph_degree = int(getattr(cfg, "cagra_graph_degree", 32))
                cagra_algo = str(getattr(cfg, "cagra_algo", "auto"))
                index_params = cagra.IndexParams(
                    metric=metric_name, graph_degree=graph_degree
                )
                search_params = cagra.SearchParams(
                    itopk_size=int(getattr(cfg, "cagra_itopk_size", 64)),
                    search_width=int(getattr(cfg, "cagra_search_width", 1)),
                    algo=cagra_algo,
                )
                self._faiss_index = cagra.build(index_params, dataset_cp)
                self._cuvs_mode = "cagra"
                self._cuvs_search_params = search_params
            else:
                raise ValueError(
                    f"Unsupported ANN index_type for cuVS: {cfg.index_type}"
                )
            self._cuvs_modules = (cp, brute_force, ivf_flat, ivf_pq, cagra)
        else:
            raise ValueError(f"Unsupported ANN backend: {backend}")

        self._faiss_index_key = tuple(features.shape)  # type: ignore
        logger.debug(
            f"ANN index ready backend={backend} index_type={cfg.index_type} "
            f"entities={num_entities} dim={dim}"
        )

    def _score_faiss_candidates(self, z_h, f_h, r, k):
        """Execute score faiss candidates.



        Args:

            z_h: Input value used by this callable.

            f_h: Input value used by this callable.

            r: Input value used by this callable.

            k: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

        if self._faiss_index is None:
            raise RuntimeError("FAISS index not initialized")
        if self._all_entity_communities is None or self._all_entity_features is None:
            raise RuntimeError("Entity latents not precomputed")
        feat_np = f_h.detach().cpu().numpy().astype("float32")
        if getattr(self, "_faiss_normalize", False):
            feat_norms = np.linalg.norm(feat_np, axis=1, keepdims=True)
            np.divide(feat_np, np.maximum(feat_norms, 1e-12), out=feat_np)
        k_eff = min(int(k), int(self._all_entity_features.shape[0]))
        backend = getattr(self, "_ann_backend", "faiss")
        if backend == "cuvs":
            if self._cuvs_modules is None or self._cuvs_mode is None:
                raise RuntimeError("cuVS backend not initialized")
            cp, brute_force, ivf_flat, ivf_pq, cagra = self._cuvs_modules
            query_cp = cp.asarray(feat_np)
            if self._cuvs_mode == "brute_force":
                _, neighbors = brute_force.search(self._faiss_index, query_cp, k_eff)
            elif self._cuvs_mode == "ivf_flat":
                if self._cuvs_search_params is None:
                    raise RuntimeError("cuVS IVF search params missing")
                _, neighbors = ivf_flat.search(
                    self._cuvs_search_params,
                    self._faiss_index,
                    query_cp,
                    k_eff,
                )
            elif self._cuvs_mode == "ivf_pq":
                if self._cuvs_search_params is None:
                    raise RuntimeError("cuVS IVFPQ search params missing")
                _, neighbors = ivf_pq.search(
                    self._cuvs_search_params,
                    self._faiss_index,
                    query_cp,
                    k_eff,
                )
            else:
                if self._cuvs_search_params is None:
                    raise RuntimeError("cuVS CAGRA search params missing")
                _, neighbors = cagra.search(
                    self._cuvs_search_params,
                    self._faiss_index,
                    query_cp,
                    k_eff,
                )
            indices = cp.asnumpy(neighbors)
        elif backend == "scann":
            try:
                result = self._faiss_index.search_batched(
                    feat_np, final_num_neighbors=k_eff
                )
            except TypeError:
                result = self._faiss_index.search_batched(feat_np, k_eff)
            if isinstance(result, tuple):
                indices = np.asarray(result[0], dtype=np.int64)
            else:
                indices = np.asarray(result, dtype=np.int64)
        else:
            _, indices = self._faiss_index.search(feat_np, k_eff)
        cand_idx = torch.from_numpy(indices).to(z_h.device)
        batch_size, num_cand = cand_idx.shape

        z_h_rep = z_h.unsqueeze(1).expand(-1, num_cand, -1).reshape(-1, z_h.shape[-1])
        f_h_rep = f_h.unsqueeze(1).expand(-1, num_cand, -1).reshape(-1, f_h.shape[-1])
        r_rep = r.unsqueeze(1).expand(-1, num_cand).reshape(-1)

        cand_z = self._all_entity_communities[cand_idx.reshape(-1)]
        cand_f = self._all_entity_features[cand_idx.reshape(-1)]

        scores = self.decoder.forward(
            z_head=z_h_rep,
            z_tail=cand_z,
            f_head=f_h_rep,
            f_tail=cand_f,
            relations=r_rep,
        ).view(batch_size, num_cand)

        if self.pc_model and self.config.lambda_pc > 0:
            pc_log = self._pc_log_prob_pairwise(z_h_rep, cand_z)
            if pc_log is not None:
                scores = scores + self.config.lambda_pc * pc_log.view(
                    batch_size, num_cand
                )

        return scores, cand_idx

    def precompute_entity_latents(self, batch_size=512):
        """Execute precompute entity latents.



        Args:

            batch_size: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.eval()
        device = self.entity_embedding.weight.device
        self._all_entity_features = torch.empty(
            self.config.num_entities, self.config.feature_dim, device=device
        )
        self._all_entity_communities = torch.empty(
            self.config.num_entities, self.config.max_communities, device=device
        )
        with torch.no_grad():
            for start in range(0, self.config.num_entities, batch_size):
                end = min(start + batch_size, self.config.num_entities)
                latents = self.encode_entities(torch.arange(start, end, device=device))
                self._all_entity_features[start:end] = latents["features"]
                self._all_entity_communities[start:end] = latents["communities"]

    def export_inference_model(
        self,
        example_triples: torch.Tensor,
        dynamic_batch: bool = True,
    ) -> torch.export.ExportedProgram:
        """Export the scoring model for production inference.

        Captures the `score_triples_batch` method into a static graph.
        Precomputes and freezes BERT embeddings if active.

        Args:
            example_triples: Sample input tensor (Batch, 3)
            dynamic_batch: Whether to allow variable batch size

        Returns:
            ExportedProgram ready for saving or AOT compilation
        """
        device = self.entity_embedding.weight.device
        if self.use_bert_relations and self._precomputed_relation_emb is None:
            logger.info("Precomputing BERT embeddings for export...")
            self.precompute_relation_embeddings(device)

        class InferenceWrapper(nn.Module):
            """Represent InferenceWrapper."""

            def __init__(self, model: DSLFMKGCModel):
                """Execute init.



                Args:

                    model: Input value used by this callable.

                """

                super().__init__()
                self.model = model

            def forward(self, triples: torch.Tensor) -> torch.Tensor:
                """Execute forward.



                Args:

                    triples: Input value used by this callable.



                Returns:

                    Return value produced by the callable.



                Notes:

                    Keep behavior deterministic and free of hidden side effects.

                """

                return self.model.score_triples_batch(triples)

        wrapper = InferenceWrapper(self).eval()

        dynamic_shapes = None
        if dynamic_batch:
            batch_dim = torch.export.Dim("batch", min=1, max=8192)
            dynamic_shapes = ({0: batch_dim, 1: 3},)

        return torch.export.export(
            wrapper, (example_triples,), dynamic_shapes=dynamic_shapes
        )

    def _heuristic_triton_threshold(self) -> int:
        return 1024


DSLFMModel = DSLFMKGCModel


def create_dslfm_kgc_model(
    config: DSLFMKGCConfig, relation_names: list[str] | None = None
) -> DSLFMKGCModel:
    """Execute create dslfm kgc model.



    Args:

        config: Input value used by this callable.

        relation_names: Optional input value.



    Returns:

        Return value produced by the callable.

    """

    return DSLFMKGCModel(config, relation_names=relation_names)


def load_dslfm_kgc_settings(
    file_manager: FileManager, path: str | Path | None = None
) -> dict[str, Any]:
    """Execute load dslfm kgc settings.



    Args:

        file_manager: Input value used by this callable.

        path: Optional input value.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    from pff.shared.core.config import DSLFM_CONFIG_PATH

    cfg_path = Path(path) if path else DSLFM_CONFIG_PATH
    if not file_manager.exists(cfg_path):
        return {}
    settings = file_manager.read(cfg_path, return_native=True)
    return settings if isinstance(settings, dict) else {}

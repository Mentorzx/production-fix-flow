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

import torch
import torch.nn as nn
import torch.nn.functional as F

from pff.shared.system.cuda import is_cuda_available

if is_cuda_available():
    try:
        import triton
        import triton.language as tl

        TRITON_EVAL_AVAILABLE = True
    except ImportError:
        TRITON_EVAL_AVAILABLE = False
        triton = None
        tl = None
else:
    TRITON_EVAL_AVAILABLE = False
    triton = None
    tl = None

from .decoder_port import DecoderStrategy
from pff.domain.learning.dslfm.triton_kernels import (
    TritonDotProductValidator,
)
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger

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
        raise NotImplementedError


class _TorchEvalBackend(_BaseEvalBackend):
    def compute_ranks(
        self,
        scores: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        true_scores = scores.gather(1, tails.unsqueeze(1))
        return (scores > true_scores).sum(dim=1) + 1


class _TritonEvalBackend(_BaseEvalBackend):
    def compute_ranks(
        self,
        scores: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ranks using a Triton streaming kernel.

        The Triton rank kernel reads tail indices as int32, so `tails` is
        converted to int32 before invocation.
        """
        if not TRITON_EVAL_AVAILABLE:
            raise RuntimeError("Triton evaluation backend is not available")
        if not scores.is_cuda:
            raise RuntimeError("Triton evaluation requires CUDA scores")
        if scores.dtype != torch.float32:
            raise RuntimeError("Triton evaluation requires float32 scores")
        if not scores.is_contiguous():
            raise RuntimeError("Triton evaluation requires contiguous scores")

        num_entities = scores.shape[1]
        ranks_out = torch.empty(
            scores.shape[0],
            device=scores.device,
            dtype=torch.int32,
        )
        tails_i32 = tails.to(torch.int32)
        block_n = 1024 if num_entities >= 1024 else 512
        grid = (scores.shape[0],)
        _rank_from_scores[grid](
            scores,
            tails_i32,
            ranks_out,
            NUM_ENTITIES=num_entities,
            BLOCK_N=block_n,
        )
        return ranks_out


if TRITON_EVAL_AVAILABLE and triton is not None and tl is not None:

    @triton.jit
    def _rank_from_scores(
        scores_ptr: Any,
        tails_ptr: Any,
        ranks_ptr: Any,
        NUM_ENTITIES: Any,
        BLOCK_N: Any,
    ):
        """Streaming rank kernel for precomputed score matrices."""
        pid = tl.program_id(0)

        offs = tl.arange(0, BLOCK_N)
        tail_idx = tl.load(tails_ptr + pid)

        true_score = tl.load(scores_ptr + pid * NUM_ENTITIES + tail_idx)

        rank_acc = tl.full((), 1, tl.int32)
        for start in range(0, NUM_ENTITIES, BLOCK_N):
            idx = start + offs
            mask = idx < NUM_ENTITIES
            block_scores = tl.load(
                scores_ptr + pid * NUM_ENTITIES + idx,
                mask=mask,
                other=-float("inf"),
            )
            better = (block_scores > true_score) & mask
            rank_acc = rank_acc + tl.sum(better.to(tl.int32), axis=0)

        tl.store(ranks_ptr + pid, rank_acc)

else:
    _rank_from_scores = None


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
    def __init__(
        self,
        config: DSLFMKGCConfig,
        relation_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.eval_backend: _BaseEvalBackend = (
            _TritonEvalBackend() if TRITON_EVAL_AVAILABLE else _TorchEvalBackend()
        )

        self.entity_embedding = nn.Embedding(
            config.num_entities,
            config.entity_dim,
        )

        self.use_bert_relations = config.use_bert_relations and relation_names is not None

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
                    logger.info(f"Encoder BERT para relacoes ativado: {config.bert_model}")
                except Exception as e:
                    logger.warning(f"Failed to load BERT relation encoder: {e}")
                    self.use_bert_relations = False
            else:
                logger.warning("transformers not available, falling back to learned embeddings")
                self.use_bert_relations = False

        if not self.use_bert_relations:
            self.relation_embedding = nn.Embedding(config.num_relations, config.entity_dim)

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
        super().train(mode)
        if mode and not self.config.stochastic_latents:
            self.vae_encoder.eval()
        return self

    def precompute_relation_embeddings(self, device: torch.device) -> None:
        if self.use_bert_relations and self._precomputed_relation_emb is None:
            if self.relation_names is None:
                logger.warning("relation_names is None, cannot precompute BERT embeddings")
                return
            self._precomputed_relation_emb = self.relation_encoder.precompute_relation_embeddings(
                self.relation_names, device
            )

    def get_relation_embeddings(self, relation_ids: torch.Tensor) -> torch.Tensor:
        if self.use_bert_relations:
            if self._precomputed_relation_emb is None:
                self.precompute_relation_embeddings(relation_ids.device)
            if self._precomputed_relation_emb is not None:
                return self._precomputed_relation_emb[relation_ids]
        return self.relation_embedding(relation_ids)

    @property
    def effective_temperature(self) -> torch.Tensor:
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
            relations = torch.zeros(heads.shape[0], dtype=torch.long, device=heads.device)
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
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        if not self.use_bert_relations:
            nn.init.xavier_uniform_(self.relation_embedding.weight)

    def encode_entities(
        self, entity_ids: torch.Tensor, temperature: float | None = None
    ) -> dict[str, torch.Tensor]:
        if temperature is None:
            temperature = self.config.temperature
        entity_emb = self.entity_embedding(entity_ids)
        return self.vae_encoder(entity_emb, temperature=temperature)

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        return_latents: bool = False,
        use_pc: bool = True,
    ) -> dict[str, torch.Tensor]:
        effective_use_pc = bool(use_pc and self.pc_model is not None and self.config.lambda_pc > 0)
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
        result = {
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
        all_ents = torch.cat([heads, tails])
        unique_ents, inverse_indices = torch.unique(all_ents, return_inverse=True)

        unique_latents = self.encode_entities(unique_ents, temperature=temperature)
        z_unique = unique_latents["communities"]
        f_unique = unique_latents["features"]

        n_triples = heads.shape[0]
        head_indices = inverse_indices[:n_triples]
        tail_indices = inverse_indices[n_triples:]

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
        batch_size = heads.shape[0]
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
            pos_scores, neg_scores, neg_ids = self.negative_sampler.get_positive_negative_scores(
                all_scores, tails, known_positive_mask=known_positive_mask
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
                z_tail_neg = neg_tail_latents["communities"].view(batch_size, global_neg_k, -1)
                f_tail_neg = neg_tail_latents["features"].view(batch_size, global_neg_k, -1)

                z_head = head_latents["communities"]
                f_head = head_latents["features"]

                z_head_rep = (
                    z_head.unsqueeze(1).expand(-1, global_neg_k, -1).reshape(-1, z_head.shape[-1])
                )
                f_head_rep = (
                    f_head.unsqueeze(1).expand(-1, global_neg_k, -1).reshape(-1, f_head.shape[-1])
                )
                relations_rep = relations.unsqueeze(1).expand(-1, global_neg_k).reshape(-1)

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
                from pff.domain.learning.dslfm.triton_kernels import (
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
            head_latents["communities"].abs().mean() + tail_latents["communities"].abs().mean()
        ) / 2

        logic_penalty = self.logic_encoder(attr_probs).mean() if self.logic_encoder else None
        pc_penalty = (
            self.pc_model(
                attr_probs,
                torch.ones(attr_probs.size(0), device=attr_probs.device, dtype=torch.long),
            ).mean()
            if self.pc_model
            else None
        )

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
            "pc_penalty": pc_penalty
            if pc_penalty is not None
            else torch.tensor(0.0, device=total_loss.device),
        }

    def _score_all_pairs(self, head_latents, tail_latents, relations, use_pc=True):
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
        if not self.pc_model:
            return None
        eps = self.config.smoothing_epsilon
        num_heads = z_heads.shape[0]
        num_tails = all_z.shape[0]
        device = z_heads.device

        if device.type == "cuda":
            try:
                from pff.domain.learning.pc.triton_kernels import (
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
                        self.pc_model.log_prior[1].item(),
                    )
                    results.append(chunk_res)
                return torch.cat(results, dim=0)
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

        return torch.cat(head_results, dim=0)

    def _pc_log_prob_pairwise(self, z_heads, z_tails):
        if not self.pc_model:
            return None
        eps = self.config.smoothing_epsilon
        combined = torch.clamp(0.5 * (z_heads + z_tails), eps, 1.0 - eps)
        attr_probs = torch.stack([combined, 1.0 - combined], dim=-1)
        labels = torch.ones(z_heads.shape[0], device=z_heads.device, dtype=torch.long)
        return self.pc_model.log_prob(attr_probs, labels)

    @torch.no_grad()
    def evaluate(
        self,
        eval_triples: torch.Tensor,
        batch_size: int = 512,
        filter_fn: (
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None
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
        num_entities = self.config.num_entities

        if self._all_entity_features is None or kwargs.get("refresh_cache", False):
            self.precompute_entity_latents(batch_size=batch_size)

        all_z = self._all_entity_communities
        all_f = self._all_entity_features

        assert all_z is not None, "Entity communities must be precomputed"
        assert all_f is not None, "Entity features must be precomputed"

        if use_faiss_eval:
            self._ensure_faiss_index(all_f)

        ranks = []
        for i in range(0, len(eval_triples), batch_size):
            batch = eval_triples[i : i + batch_size].to(device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            head_latents = self.encode_entities(h)
            z_h, f_h = head_latents["communities"], head_latents["features"]

            true_scores = self.decoder.forward(
                z_head=z_h, z_tail=all_z[t], f_head=f_h, f_tail=all_f[t], relations=r
            )
            if self.pc_model and self.config.lambda_pc > 0:
                pc_log_true = self._pc_log_prob_pairwise(z_h, all_z[t])
                if pc_log_true is not None:
                    true_scores = true_scores + self.config.lambda_pc * pc_log_true

            batch_ranks = torch.ones(len(h), device=device, dtype=torch.int32)

            if use_faiss_eval:
                cand_scores, cand_idx = self._score_faiss_candidates(z_h, f_h, r, faiss_candidate_k)
                if filter_fn:
                    cand_scores = filter_fn(cand_scores, h, r, cand_idx)
                better_in_cand = (cand_scores > true_scores.unsqueeze(1)).sum(dim=1)
                batch_ranks += better_in_cand.to(torch.int32)
            else:
                can_use_triton = (
                    TRITON_EVAL_AVAILABLE
                    and device.type == "cuda"
                    and isinstance(self.decoder, StochasticBlockmodelDecoder)
                    and (not self.pc_model or self.config.lambda_pc <= 0)
                    and filter_fn is None
                )

                if can_use_triton:
                    pass

                if can_use_triton and not hasattr(self, "_triton_validator_cache"):
                    sbm_decoder: Any = self.decoder
                    w_c = torch.sqrt(torch.abs(sbm_decoder.community_weight))
                    w_f = torch.sqrt(torch.abs(sbm_decoder.feature_weight))

                    e_c = all_z * w_c

                    f_norm = F.normalize(all_f, p=2, dim=-1)
                    if sbm_decoder.use_bilinear:
                        e_f = f_norm * w_f
                    else:
                        e_f = f_norm * w_f

                    e_b = torch.ones(num_entities, 1, device=device, dtype=all_z.dtype)

                    entities_proj = torch.cat([e_c, e_f, e_b], dim=1)
                    self._triton_validator_cache = TritonDotProductValidator(
                        entities_proj, device=str(device)
                    )

                if can_use_triton and hasattr(self, "_triton_validator_cache"):
                    sbm_decoder: Any = self.decoder
                    w_c = torch.sqrt(torch.abs(sbm_decoder.community_weight))
                    w_f = torch.sqrt(torch.abs(sbm_decoder.feature_weight))

                    e_c = all_z * w_c

                    f_norm = F.normalize(all_f, p=2, dim=-1)
                    if sbm_decoder.use_bilinear:
                        e_f = f_norm * w_f
                    else:
                        e_f = f_norm * w_f

                    e_b = torch.ones(num_entities, 1, device=device, dtype=all_z.dtype)

                    entities_proj = torch.cat([e_c, e_f, e_b], dim=1)
                    self._triton_validator_cache = TritonDotProductValidator(
                        entities_proj, device=str(device)
                    )

                if can_use_triton and hasattr(self, "_triton_validator_cache"):
                    sbm_decoder: Any = self.decoder
                    w_c = torch.sqrt(torch.abs(sbm_decoder.community_weight))
                    w_f = torch.sqrt(torch.abs(sbm_decoder.feature_weight))

                    W_r = sbm_decoder.W[r]  # noqa: N806
                    q_c = torch.bmm(z_h.unsqueeze(1), W_r).squeeze(1) * w_c

                    f_h_norm = F.normalize(f_h, p=2, dim=-1)
                    if sbm_decoder.use_bilinear:
                        weight = sbm_decoder.feature_bilinear.weight.squeeze(0)
                        q_f = torch.mm(f_h_norm, weight) * w_f
                    else:
                        q_f = f_h_norm * w_f

                    q_b = sbm_decoder.relation_bias[r].unsqueeze(1)

                    queries_proj = torch.cat([q_c, q_f, q_b], dim=1)

                    batch_ranks_triton = self._triton_validator_cache.compute_ranks(queries_proj, t)

                    ranks.append(batch_ranks_triton)
                    continue

                for start in range(0, num_entities, score_all_tails_chunk_size):
                    end = min(start + score_all_tails_chunk_size, num_entities)
                    chunk_z = all_z[start:end]
                    chunk_f = all_f[start:end]

                    chunk_scores = self.decoder.score_all_tails(
                        z_head=z_h,
                        f_head=f_h,
                        relations=r,
                        all_z=chunk_z,
                        all_f=chunk_f,
                    )

                    if self.pc_model and self.config.lambda_pc > 0:
                        pc_log_chunk = self._pc_log_prob_matrix(z_h, chunk_z)
                        if pc_log_chunk is not None:
                            chunk_scores = chunk_scores + self.config.lambda_pc * pc_log_chunk

                    if filter_fn is not None:
                                                             
                        if score_all_tails_chunk_size >= num_entities:
                                                                     
                            candidates = torch.arange(num_entities, device=device)
                        else:
                                                                          
                            candidates = torch.arange(start, end, device=device)

                                                                                          
                                                                                    
                        chunk_scores = filter_fn(chunk_scores, h, r, candidates, t)

                    batch_ranks += (
                        (chunk_scores > true_scores.unsqueeze(1)).sum(dim=1).to(torch.int32)
                    )

            ranks.append(batch_ranks)

        if hasattr(self, "_triton_validator_cache"):
            del self._triton_validator_cache

        all_ranks = torch.cat(ranks).float()
        mrr = (1.0 / all_ranks).mean().item()
        hits_at_1 = (all_ranks <= 1).float().mean().item()
        hits_at_3 = (all_ranks <= 3).float().mean().item()
        hits_at_10 = (all_ranks <= 10).float().mean().item()

        ap_at_10 = (1.0 / all_ranks.clamp(max=10)).mean().item()

        return {
            "mrr": mrr,
            "hits@1": hits_at_1,
            "hits@3": hits_at_3,
            "hits@10": hits_at_10,
            "ap@10": ap_at_10,
        }

    def _ensure_faiss_index(self, features: torch.Tensor) -> None:
        if self._faiss_index is not None and self._faiss_index_key == features.shape:
            return
        from pff.shared.acceleration.faiss_utils import import_faiss

        faiss_lib, available = import_faiss()
        if not available or faiss_lib is None:
            raise ImportError("faiss-cpu not installed. Run: poetry add faiss-cpu")

        feat_np = features.detach().cpu().numpy().astype("float32")
        self._faiss_index = faiss_lib.IndexFlatIP(feat_np.shape[1])
        if self._faiss_index is not None:
            self._faiss_index.add(feat_np)
        self._faiss_index_key = tuple(features.shape)  # type: ignore

    def _score_faiss_candidates(self, z_h, f_h, r, k):
        if self._faiss_index is None:
            raise RuntimeError("FAISS index not initialized")
        feat_np = f_h.detach().cpu().numpy().astype("float32")
        _, indices = self._faiss_index.search(feat_np, k)
        cand_idx = torch.from_numpy(indices).to(z_h.device)
        batch_size, num_cand = cand_idx.shape

        z_h_rep = z_h.unsqueeze(1).expand(-1, num_cand, -1).reshape(-1, z_h.shape[-1])
        f_h_rep = f_h.unsqueeze(1).expand(-1, num_cand, -1).reshape(-1, f_h.shape[-1])
        r_rep = r.unsqueeze(1).expand(-1, num_cand).reshape(-1)

        if self._all_entity_communities is None or self._all_entity_features is None:
            raise RuntimeError("Entity latents not precomputed")

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
            pc_log = self._pc_log_prob_matrix(z_h, cand_z)
            if pc_log is not None:
                scores = scores + self.config.lambda_pc * pc_log.view(batch_size, num_cand)

        return scores, cand_idx

    def precompute_entity_latents(self, batch_size=512):
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

    def _heuristic_triton_threshold(self) -> int:
        return 1024


DSLFMModel = DSLFMKGCModel


def create_dslfm_kgc_model(
    config: DSLFMKGCConfig, relation_names: list[str] | None = None
) -> DSLFMKGCModel:
    return DSLFMKGCModel(config, relation_names=relation_names)


def load_dslfm_kgc_settings(
    file_manager: FileManager, path: str | Path | None = None
) -> dict[str, Any]:
    from pff.shared.core.config import DSLFM_CONFIG_PATH

    cfg_path = Path(path) if path else DSLFM_CONFIG_PATH
    if not file_manager.exists(cfg_path):
        return {}
    settings = file_manager.read(cfg_path, return_native=True)
    return settings if isinstance(settings, dict) else {}

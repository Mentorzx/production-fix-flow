"""RotatE Core Implementation.

This module implements the RotatE (Rotation in Complex Space) model for knowledge
graph completion. RotatE represents entities as complex vectors and relations as
rotations: h ∘ r = t, where r = e^(iθ).

Design Patterns Applied:
    - **Strategy Pattern:** RotatE can be swapped with TransE via KGEModelStrategy.
    - **Template Method:** Training follows the base trainer structure.
    - **Observer Pattern:** Metrics observed via TrainingObserver infrastructure.
    - **Factory Pattern:** Model creation via _initialize_embeddings().

Mathematical Foundation:
    Given a triple (h, r, t):
    - h, t ∈ ℂ^d: entity embeddings as complex vectors
    - r = e^(iθ_r) ∈ ℂ^d: relation as rotation (phase angles θ_r ∈ [-π, π])

    Scoring function:
        score(h, r, t) = γ - ||h ∘ r - t||

    Where:
    - γ is a fixed margin (gamma)
    - ∘ is element-wise (Hadamard) product
    - || · || is the L1 or L2 norm

Performance Optimizations:
    - Batch scoring via score_triples_batch() for 10-100x speedup.
    - Vectorized complex operations using PyTorch's complex tensor support.
    - Self-adversarial negative sampling focuses on hard negatives.
    - Optional Numba kernels for CPU-bound operations.

SOTA References:
    - Sun et al. 2019 "RotatE: Knowledge Graph Embedding by Relational
      Rotation in Complex Space" (ICLR 2019)
    - PyKEEN: NSSALoss for self-adversarial training

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pff.utils import logger
from pff.validators.rotate.config import RotatEConfig


class RotatEModel(nn.Module):
    """RotatE model implementation with complex embeddings.

    RotatE models relations as rotations in complex vector space:
        h ∘ r = t
    Where r = e^(iθ) represents rotation by angle θ.

    Attributes:
        num_entities: Number of entities in the KG.
        num_relations: Number of relations in the KG.
        embedding_dim: Full embedding dimension (2 * complex_dim).
        gamma: Fixed margin for scoring.
        epsilon: Modulus regularization parameter.
        entity_embedding: Entity embeddings stored as [real, imag] pairs.
        relation_embedding: Relation phase angles θ ∈ [-π, π].

    Example:
        >>> model = RotatEModel(num_entities=5000, num_relations=50)
        >>> scores = model.score_triples_batch(triples)
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        gamma: float = 12.0,
        epsilon: float = 2.0,
        config: RotatEConfig | dict[str, Any] | None = None,
    ) -> None:
        """Initialize RotatE model.

        Args:
            num_entities: Number of entities in the knowledge graph.
            num_relations: Number of relations in the knowledge graph.
            embedding_dim: Dimension of embeddings (must be even).
            gamma: Fixed margin for scoring function.
            epsilon: Modulus regularization parameter.
            config: Additional configuration (RotatEConfig or dict).

        Raises:
            ValueError: If embedding_dim is odd.
        """
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(
                f"embedding_dim must be even for complex representation, "
                f"got {embedding_dim}"
            )

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.complex_dim = embedding_dim // 2
        self.gamma = gamma
        self.epsilon = epsilon

        if isinstance(config, RotatEConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = RotatEConfig(**{
                k: v for k, v in config.items()
                if k in RotatEConfig.__dataclass_fields__
            })
        else:
            self.config = RotatEConfig(
                embedding_dim=embedding_dim,
                gamma=gamma,
                epsilon=epsilon,
            )

        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, self.complex_dim)
        self._embedding_range = (gamma + epsilon) / embedding_dim
        self._initialize_embeddings()
        # LRU cache com limite de 5000 entradas (OrderedDict para política LRU)
        self._score_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._score_cache_maxsize: int = 5000

        logger.info(
            f"RotatE inicializado: {num_entities:,} entidades, "
            f"{num_relations} relacoes, dim={embedding_dim}, gamma={gamma}"
        )

    def _initialize_embeddings(self) -> None:
        """Initialize embeddings following RotatE paper.

        Entity embeddings: Uniform in [-embedding_range, embedding_range]
        Relation embeddings: Uniform in [-π, π] (phase angles)
        """
        nn.init.uniform_(
            self.entity_embedding.weight.data,
            -self._embedding_range,
            self._embedding_range,
        )
        nn.init.uniform_(
            self.relation_embedding.weight.data,
            -math.pi,
            math.pi,
        )

    def _split_complex(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split embedding into real and imaginary parts.

        Args:
            embedding: Tensor of shape [..., embedding_dim].

        Returns:
            Tuple of (real_part, imag_part) each with shape [..., complex_dim].
        """
        return embedding[..., : self.complex_dim], embedding[..., self.complex_dim :]

    def _complex_multiply(
        self,
        re_a: torch.Tensor,
        im_a: torch.Tensor,
        re_b: torch.Tensor,
        im_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform complex multiplication (a * b) in real representation.

        (a_re + i*a_im) * (b_re + i*b_im) = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)

        Args:
            re_a: Real part of first operand.
            im_a: Imaginary part of first operand.
            re_b: Real part of second operand.
            im_b: Imaginary part of second operand.

        Returns:
            Tuple of (real_result, imag_result).
        """
        re_result = re_a * re_b - im_a * im_b
        im_result = re_a * im_b + im_a * re_b
        return re_result, im_result

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        mode: str = "single",
    ) -> torch.Tensor:
        """Forward pass computing RotatE scores.

        Computes: score = γ - ||h ∘ r - t||

        Where h ∘ r is the rotation of h by relation r = e^(iθ).

        Args:
            heads: Head entity indices [batch_size] or [batch_size, 1].
            relations: Relation indices [batch_size] or [batch_size, 1].
            tails: Tail entity indices [batch_size] or [batch_size, num_neg].
            mode: Scoring mode ('single', 'head-batch', 'tail-batch').

        Returns:
            Scores tensor. Higher scores indicate more plausible triples.
        """
        heads = torch.clamp(heads.long(), 0, self.num_entities - 1)
        relations = torch.clamp(relations.long(), 0, self.num_relations - 1)
        tails = torch.clamp(tails.long(), 0, self.num_entities - 1)
        head_emb = self.entity_embedding(heads)
        tail_emb = self.entity_embedding(tails)
        phase = self.relation_embedding(relations)
        re_h, im_h = self._split_complex(head_emb)
        re_t, im_t = self._split_complex(tail_emb)
        re_r = torch.cos(phase)
        im_r = torch.sin(phase)

        if mode == "head-batch":
            re_r = re_r.unsqueeze(1)
            im_r = im_r.unsqueeze(1)
            re_t = re_t.unsqueeze(1)
            im_t = im_t.unsqueeze(1)
        elif mode == "tail-batch":
            re_h = re_h.unsqueeze(1)
            im_h = im_h.unsqueeze(1)
            re_r = re_r.unsqueeze(1)
            im_r = im_r.unsqueeze(1)
        re_rot, im_rot = self._complex_multiply(re_h, im_h, re_r, im_r)
        re_diff = re_rot - re_t
        im_diff = im_rot - im_t
        distance = torch.sqrt(re_diff ** 2 + im_diff ** 2 + 1e-12)
        distance_norm = distance.sum(dim=-1)
        score = self.gamma - distance_norm

        return score

    def score_triple(self, head_idx: int, rel_idx: int, tail_idx: int) -> float:
        """Score a single triple.

        For batch operations, use score_triples_batch() for 10-100x speedup.

        Args:
            head_idx: Head entity index.
            rel_idx: Relation index.
            tail_idx: Tail entity index.

        Returns:
            Score for the triple (higher = more plausible).
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            heads = torch.tensor([head_idx], dtype=torch.long, device=device)
            relations = torch.tensor([rel_idx], dtype=torch.long, device=device)
            tails = torch.tensor([tail_idx], dtype=torch.long, device=device)
            score = self.forward(heads, relations, tails)
            return score.item()

    def score_triples_batch(
        self,
        triples: np.ndarray | torch.Tensor,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Score multiple triples efficiently using batch processing.

        This is 10-100x faster than calling score_triple() in a loop.
        Uses vectorized GPU operations when available.

        Args:
            triples: Array of shape (n_triples, 3) with [head, rel, tail] indices.
            use_cache: If True, caches results for repeated queries.

        Returns:
            Array of scores with shape (n_triples,).

        Example:
            >>> triples = np.array([[0, 1, 2], [3, 4, 5]])
            >>> scores = model.score_triples_batch(triples)
        """
        if isinstance(triples, torch.Tensor):
            triples_arr = triples.cpu().numpy() if triples.is_cuda else triples.numpy()
        else:
            triples_arr = triples

        cache_key = None
        if use_cache:
            from pff.utils.hash import stable_hash
            cache_key = stable_hash(triples_arr.tobytes())
            if cache_key in self._score_cache:
                # Move para o final (mais recente usado) para política LRU
                self._score_cache.move_to_end(cache_key)
                return self._score_cache[cache_key]

        with torch.no_grad():
            device = next(self.parameters()).device

            if isinstance(triples, np.ndarray):
                triples_t = torch.from_numpy(triples).long().to(device)
            else:
                triples_t = triples.long().to(device)

            heads = triples_t[:, 0]
            relations = triples_t[:, 1]
            tails = triples_t[:, 2]

            scores = self.forward(heads, relations, tails)
            result = scores.cpu().numpy()

        if use_cache and cache_key is not None:
            # Política LRU: remove entradas mais antigas quando atinge o limite
            while len(self._score_cache) >= self._score_cache_maxsize:
                self._score_cache.popitem(last=False)  # Remove mais antigo (FIFO)
            self._score_cache[cache_key] = result

        return result

    def clear_score_cache(self) -> None:
        """Clear the scoring cache to free memory."""
        self._score_cache.clear()

    def compute_loss(
        self,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor,
        subsampling_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute self-adversarial negative sampling loss.

        Loss = -log σ(γ - d_r(h,t)) - Σ p_i * log σ(d_r(h'_i,t'_i) - γ)

        Where p_i is the self-adversarial weight for negative i.

        Args:
            positive_triples: Positive samples [batch_size, 3].
            negative_triples: Negative samples [batch_size, num_neg, 3].
            subsampling_weight: Optional per-sample weights [batch_size].

        Returns:
            Scalar loss tensor.
        """
        batch_size, num_neg, _ = negative_triples.shape
        pos_scores = self.forward(
            positive_triples[:, 0],
            positive_triples[:, 1],
            positive_triples[:, 2],
        )
        # Vectorized negative scoring: reshape to process all negatives at once
        # Shape: [batch_size * num_neg, 3] -> forward -> [batch_size, num_neg]
        neg_flat = negative_triples.view(-1, 3)
        neg_scores_flat = self.forward(neg_flat[:, 0], neg_flat[:, 1], neg_flat[:, 2])
        neg_scores = neg_scores_flat.view(batch_size, num_neg)
        
        pos_loss = -F.logsigmoid(pos_scores)

        if self.config.use_self_adversarial:
            neg_probs = F.softmax(
                neg_scores * self.config.adversarial_temperature, dim=1
            ).detach()
            neg_loss = -(neg_probs * F.logsigmoid(-neg_scores)).sum(dim=1)
        else:
            neg_loss = -F.logsigmoid(-neg_scores).mean(dim=1)
        loss = pos_loss + neg_loss

        if subsampling_weight is not None:
            loss = loss * subsampling_weight

        return loss.mean()

    def get_entity_embeddings(
        self, entity_ids: torch.Tensor | list[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get entity embeddings as (real, imaginary) parts.

        Args:
            entity_ids: Entity indices. If None, returns all embeddings.

        Returns:
            Tuple of (real_embeddings, imag_embeddings).
        """
        if entity_ids is None:
            emb = self.entity_embedding.weight
        else:
            if isinstance(entity_ids, list):
                entity_ids = torch.tensor(entity_ids, dtype=torch.long)
            device = next(self.parameters()).device
            entity_ids = entity_ids.to(device)
            emb = self.entity_embedding(entity_ids)

        return self._split_complex(emb)

    def get_relation_phases(
        self, relation_ids: torch.Tensor | list[int] | None = None
    ) -> torch.Tensor:
        """Get relation phase angles.

        Args:
            relation_ids: Relation indices. If None, returns all phases.

        Returns:
            Phase angles tensor with shape [num_relations, complex_dim].
        """
        if relation_ids is None:
            return self.relation_embedding.weight

        if isinstance(relation_ids, list):
            relation_ids = torch.tensor(relation_ids, dtype=torch.long)

        device = next(self.parameters()).device
        relation_ids = relation_ids.to(device)
        return self.relation_embedding(relation_ids)

    def normalize_embeddings(self) -> None:
        """Normalize entity embeddings (optional for RotatE).

        Note: RotatE doesn't strictly require normalization like TransE,
        but this can help stabilize training in some cases.
        """
        pass

    def regularization_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss for embeddings.

        Returns:
            Regularization loss tensor.
        """
        entity_reg = self.config.entity_regularizer_weight
        relation_reg = self.config.relation_regularizer_weight

        if entity_reg <= 0 and relation_reg <= 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        loss = torch.tensor(0.0, device=next(self.parameters()).device)

        if entity_reg > 0:
            loss = loss + entity_reg * self.entity_embedding.weight.norm(p=2) ** 2

        if relation_reg > 0:
            loss = loss + relation_reg * self.relation_embedding.weight.norm(p=2) ** 2

        return loss

    def get_embedding_stats(self) -> dict[str, float]:
        """Get statistics about embeddings for monitoring.

        Returns:
            Dictionary with embedding statistics.
        """
        with torch.no_grad():
            entity_emb = self.entity_embedding.weight
            relation_phases = self.relation_embedding.weight

            re_ent, im_ent = self._split_complex(entity_emb)
            entity_magnitude = torch.sqrt(re_ent ** 2 + im_ent ** 2).mean()

            return {
                "entity_embedding_mean": entity_emb.mean().item(),
                "entity_embedding_std": entity_emb.std().item(),
                "entity_magnitude_mean": entity_magnitude.item(),
                "relation_phase_mean": relation_phases.mean().item(),
                "relation_phase_std": relation_phases.std().item(),
                "relation_phase_min": relation_phases.min().item(),
                "relation_phase_max": relation_phases.max().item(),
            }


class RotatEDataset(torch.utils.data.Dataset):
    """Dataset for RotatE training with negative sampling.

    Generates negative samples by corrupting either head or tail entities.

    Attributes:
        triples: Training triples as torch tensor.
        num_entities: Total number of entities.
        num_negatives: Number of negative samples per positive.
        rng: Random number generator for reproducibility.
    """

    def __init__(
        self,
        triples: np.ndarray,
        num_entities: int,
        num_negatives: int = 256,
        seed: int = 42,
    ) -> None:
        """Initialize dataset.

        Args:
            triples: Array of triples [num_triples, 3].
            num_entities: Total number of entities.
            num_negatives: Number of negative samples per positive.
            seed: Random seed for reproducibility.
        """
        self.triples = torch.from_numpy(triples).long()
        self.num_entities = num_entities
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return number of triples."""
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample with negative sampling.

        Args:
            idx: Index of the positive triple.

        Returns:
            Dictionary with 'positive' and 'negatives' tensors.
        """
        positive = self.triples[idx]
        head, rel, tail = positive
        
        # Vectorized negative sampling: generate all negatives at once
        corrupt_head_mask = self.rng.random(self.num_negatives) < 0.5
        random_entities = self.rng.integers(0, self.num_entities, size=self.num_negatives)
        
        # Pre-allocate negatives tensor
        negatives = positive.unsqueeze(0).expand(self.num_negatives, -1).clone()
        
        # Apply head corruption where mask is True
        negatives[corrupt_head_mask, 0] = torch.from_numpy(
            random_entities[corrupt_head_mask]
        ).long()
        # Apply tail corruption where mask is False
        negatives[~corrupt_head_mask, 2] = torch.from_numpy(
            random_entities[~corrupt_head_mask]
        ).long()

        return {
            "positive": positive,
            "negatives": negatives,
        }

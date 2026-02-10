"""Stochastic Blockmodel decoder for DSLFM-KGC.

This module implements the decoder that scores triples based on:
1. Community-level interactions (learned W matrix per relation)
2. Feature-level interactions (Hadamard product)

Design Patterns:
    - Strategy: Different scoring functions can be swapped
    - Composite: Combines community and feature scores
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .decoder_port import DecoderStrategy


class StochasticBlockmodelDecoder(nn.Module, DecoderStrategy):
    """Stochastic Blockmodel decoder for link prediction.

    Scores triples using community-based and feature-based interactions:

    score(h, r, t) = z_h @ W_r @ z_t^T + f_h · f_t + bias_r

    Design Patterns:
        - Strategy: Implements DecoderStrategy protocol.
    """

    def __init__(
        self,
        num_communities: int,
        feature_dim: int,
        num_relations: int,
        community_weight: float = 1.0,
        feature_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.num_communities = num_communities
        self.feature_dim = feature_dim
        self.num_relations = num_relations

        self.W = nn.Parameter(torch.zeros(num_relations, num_communities, num_communities))

        self.relation_bias = nn.Parameter(torch.zeros(num_relations))

        self.community_weight = nn.Parameter(torch.tensor(community_weight))
        self.feature_weight = nn.Parameter(torch.tensor(feature_weight))

        self.use_bilinear = feature_dim <= 128
        if self.use_bilinear:
            self.feature_bilinear = nn.Bilinear(feature_dim, feature_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training.

        Initializes `W` near a diagonal structure to encourage within-community
        links early in training.
        """
        nn.init.normal_(self.W, mean=0.0, std=0.01)

        eye = torch.eye(self.num_communities, device=self.W.data.device)
        self.W.data.add_(0.1 * eye.unsqueeze(0))

    def community_score(
        self,
        z_head: torch.Tensor,
        z_tail: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute community-based interaction score efficiently."""
        z_h_W = torch.bmm(z_head.unsqueeze(1), self.W[relations]).squeeze(1)  # noqa: N806
        return (z_h_W * z_tail).sum(dim=-1)

    def feature_score(
        self,
        f_head: torch.Tensor,
        f_tail: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature-based interaction score.

        Args:
            f_head: Head feature vectors [batch, feature_dim].
            f_tail: Tail feature vectors [batch, feature_dim].

        Returns:
            Feature scores [batch].
        """
        if self.use_bilinear:
            return self.feature_bilinear(f_head, f_tail).squeeze(-1)

        return (f_head * f_tail).sum(dim=-1)

    def forward(
        self,
        z_head: torch.Tensor,
        z_tail: torch.Tensor,
        f_head: torch.Tensor,
        f_tail: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        """Score triples using combined community and feature interactions."""
        c_score = self.community_score(z_head, z_tail, relations)

        f_score = self.feature_score(
            F.normalize(f_head, p=2, dim=-1), F.normalize(f_tail, p=2, dim=-1)
        )

        r_bias = self.relation_bias[relations]
        return self.community_weight * c_score + self.feature_weight * f_score + r_bias

    def score_all_tails(
        self,
        z_head: torch.Tensor,
        f_head: torch.Tensor,
        relations: torch.Tensor,
        all_z: torch.Tensor,
        all_f: torch.Tensor,
    ) -> torch.Tensor:
        """Score head against all possible tails (for ranking).

        Args:
            z_head: Head community memberships [batch, num_communities].
            f_head: Head feature vectors [batch, feature_dim].
            relations: Relation indices [batch].
            all_z: All entity community memberships [num_entities, num_communities].
            all_f: All entity feature vectors [num_entities, feature_dim].

        Returns:
            Scores for all tails [batch, num_entities].
        """
        W_r = self.W[relations]  # noqa: N806

        z_h_W = torch.bmm(z_head.unsqueeze(1), W_r).squeeze(1)  # noqa: N806

        c_scores = torch.mm(z_h_W, all_z.t())

        f_head = F.normalize(f_head, p=2, dim=-1)
        all_f = F.normalize(all_f, p=2, dim=-1)

        if self.use_bilinear:
            weight = self.feature_bilinear.weight.squeeze(0)
            f_head_trans = torch.mm(f_head, weight)
            f_scores = torch.mm(f_head_trans, all_f.t())
        else:
            f_scores = torch.mm(f_head, all_f.t())

        r_bias = self.relation_bias[relations].unsqueeze(1)

        return self.community_weight * c_scores + self.feature_weight * f_scores + r_bias

    def prepare_for_triton(
        self,
        z_head: torch.Tensor,
        f_head: torch.Tensor,
        relations: torch.Tensor,
        all_z: torch.Tensor,
        all_f: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project queries and entities into a joint space for dot-product ranking.

        SBM Score: w_c (z_h W_r z_t^T) + w_f (f_h f_t^T) + bias
        Can be rewritten as dot product Q . E where:
        Q = [sqrt(w_c) * (z_h W_r), sqrt(w_f) * f_h, bias]
        E = [sqrt(w_c) * z_t,       sqrt(w_f) * f_t, 1   ]

        Args:
            z_head: [Batch, K]
            f_head: [Batch, F]
            relations: [Batch]
            all_z: [NumEntities, K]
            all_f: [NumEntities, F]

        Returns:
            queries: [Batch, K+F+1]
            entities: [NumEntities, K+F+1]
        """
        num_entities = all_z.shape[0]

        w_c = torch.sqrt(torch.abs(self.community_weight))
        w_f = torch.sqrt(torch.abs(self.feature_weight))

        W_r = self.W[relations]  # noqa: N806

        q_c = torch.bmm(z_head.unsqueeze(1), W_r).squeeze(1) * w_c
        e_c = all_z * w_c

        f_head_norm = F.normalize(f_head, p=2, dim=-1)
        all_f_norm = F.normalize(all_f, p=2, dim=-1)

        if self.use_bilinear:
            weight = self.feature_bilinear.weight.squeeze(0)
            q_f = torch.mm(f_head_norm, weight) * w_f
            e_f = all_f_norm * w_f
        else:
            q_f = f_head_norm * w_f
            e_f = all_f_norm * w_f

        r_bias = self.relation_bias[relations].unsqueeze(1)
        q_b = r_bias
        e_b = torch.ones(num_entities, 1, device=all_z.device, dtype=all_z.dtype)

        queries = torch.cat([q_c, q_f, q_b], dim=1)
        entities = torch.cat([e_c, e_f, e_b], dim=1)

        return queries, entities


class LowRankSBMDecoder(nn.Module, DecoderStrategy):
    """Low-rank Stochastic Blockmodel decoder for reduced memory usage.

    Uses basis decomposition for community interaction matrices:
    W_r = Σ_i a_{r,i} * U_i

    Design Patterns:
        - Strategy: Implements DecoderStrategy protocol.
    """

    def __init__(
        self,
        num_communities: int,
        feature_dim: int,
        num_relations: int,
        num_basis: int = 8,
        community_weight: float = 1.0,
        feature_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.num_communities = num_communities
        self.feature_dim = feature_dim
        self.num_relations = num_relations
        self.num_basis = num_basis

        self.basis_matrices = nn.Parameter(torch.zeros(num_basis, num_communities, num_communities))

        self.relation_coeffs = nn.Parameter(torch.zeros(num_relations, num_basis))

        self.relation_bias = nn.Parameter(torch.zeros(num_relations))

        self.community_weight = nn.Parameter(torch.tensor(community_weight))
        self.feature_weight = nn.Parameter(torch.tensor(feature_weight))

        self.use_bilinear = feature_dim <= 128
        if self.use_bilinear:
            self.feature_bilinear = nn.Bilinear(feature_dim, feature_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        nn.init.orthogonal_(self.basis_matrices.view(self.num_basis, -1))
        self.basis_matrices.data = self.basis_matrices.data.view(
            self.num_basis, self.num_communities, self.num_communities
        )

        nn.init.xavier_uniform_(self.relation_coeffs)

        diag_indices = torch.arange(
            min(self.num_basis, self.num_communities),
            device=self.basis_matrices.data.device,
        )
        self.basis_matrices.data[
            diag_indices, diag_indices, diag_indices % self.num_communities
        ] += 0.1

    def get_relation_matrix(self, relations: torch.Tensor) -> torch.Tensor:
        """Compute W_r from basis decomposition.

        Args:
            relations: Relation indices [batch].

        Returns:
            W_r matrices [batch, K, K].
        """
        coeffs = self.relation_coeffs[relations]
        W_r = torch.einsum("br,rij->bij", coeffs, self.basis_matrices)
        return W_r

    def community_score(
        self,
        z_head: torch.Tensor,
        z_tail: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute community-based interaction score.

        Args:
            z_head: Head community memberships [batch, num_communities].
            z_tail: Tail community memberships [batch, num_communities].
            relations: Relation indices [batch].

        Returns:
            Community scores [batch].
        """
        W_r = self.get_relation_matrix(relations)
        score = torch.einsum("bi,bij,bj->b", z_head, W_r, z_tail)
        return score

    def feature_score(
        self,
        f_head: torch.Tensor,
        f_tail: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature-based interaction score."""
        if self.use_bilinear:
            return self.feature_bilinear(f_head, f_tail).squeeze(-1)
        return (f_head * f_tail).sum(dim=-1)

    def forward(
        self,
        z_head: torch.Tensor,
        z_tail: torch.Tensor,
        f_head: torch.Tensor,
        f_tail: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        """Score triples combining community and feature interactions.

        Args:
            z_head: Head community memberships [batch, num_communities].
            z_tail: Tail community memberships [batch, num_communities].
            f_head: Head feature vectors [batch, feature_dim].
            f_tail: Tail feature vectors [batch, feature_dim].
            relations: Relation indices [batch].

        Returns:
            Triple scores [batch].
        """
        c_score = self.community_score(z_head, z_tail, relations)
        f_score = self.feature_score(f_head, f_tail)

        return (
            self.community_weight * c_score
            + self.feature_weight * f_score
            + self.relation_bias[relations]
        )

    def score_all_tails(
        self,
        z_head: torch.Tensor,
        f_head: torch.Tensor,
        relations: torch.Tensor,
        all_z: torch.Tensor,
        all_f: torch.Tensor,
    ) -> torch.Tensor:
        """Score head against all possible tails (for ranking)."""
        W_r = self.get_relation_matrix(relations)
        z_h_W = torch.bmm(z_head.unsqueeze(1), W_r).squeeze(1)
        c_scores = torch.mm(z_h_W, all_z.t())

        f_head = F.normalize(f_head, p=2, dim=-1)
        all_f = F.normalize(all_f, p=2, dim=-1)

        if self.use_bilinear:
            weight = self.feature_bilinear.weight.squeeze(0)
            f_head_trans = torch.mm(f_head, weight)
            f_scores = torch.mm(f_head_trans, all_f.t())
        else:
            f_scores = torch.mm(f_head, all_f.t())

        r_bias = self.relation_bias[relations].unsqueeze(1)

        return self.community_weight * c_scores + self.feature_weight * f_scores + r_bias

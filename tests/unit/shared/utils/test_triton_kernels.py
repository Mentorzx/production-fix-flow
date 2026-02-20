"""
Tests for Triton DSLFM validation kernels.

Verifies correctness by comparing Triton ranks against PyTorch reference.
"""

import pytest
import torch

from pff.shared.acceleration.triton_kernels import (
    TritonDSLFMValidator,
    is_triton_available,
    pc2_forward_triton,
    pc2_matrix_forward_triton,
)

# Use is_triton_available() to check for both library availability and hardware
TRITON_ACTUALLY_AVAILABLE = is_triton_available()

pytestmark = pytest.mark.skipif(
    not TRITON_ACTUALLY_AVAILABLE or not torch.cuda.is_available(),
    reason="Triton or CUDA not available",
)


def _compute_pytorch_ranks(
    query_re: torch.Tensor,
    query_im: torch.Tensor,
    entity_re: torch.Tensor,
    entity_im: torch.Tensor,
    true_tails: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Reference PyTorch implementation for rank calculation."""
    batch_size = query_re.shape[0]

    diff_re = query_re.unsqueeze(1) - entity_re.unsqueeze(0)
    diff_im = query_im.unsqueeze(1) - entity_im.unsqueeze(0)
    dist_sq = (diff_re * diff_re + diff_im * diff_im).sum(dim=-1)
    scores = gamma - torch.sqrt(dist_sq)

    true_scores = scores[torch.arange(batch_size), true_tails]
    ranks = (scores > true_scores.unsqueeze(1)).sum(dim=1) + 1

    return ranks


class TestTritonAvailability:
    """Tests for Triton availability checks."""

    def test_is_triton_available(self):
        """Execute test is triton available."""

        assert is_triton_available() is True


class TestTritonDSLFMValidator:
    """Tests for TritonDSLFMValidator correctness."""

    @pytest.fixture
    def small_embeddings(self):
        """Create small synthetic embeddings for testing."""
        torch.manual_seed(42)
        num_entities = 100
        dim = 64
        gamma = 12.0

        entity_re = torch.randn(num_entities, dim, dtype=torch.float32)
        entity_im = torch.randn(num_entities, dim, dtype=torch.float32)

        return entity_re, entity_im, gamma

    @pytest.fixture
    def medium_embeddings(self):
        """Create medium synthetic embeddings for testing."""
        torch.manual_seed(42)
        num_entities = 1000
        dim = 128
        gamma = 12.0

        entity_re = torch.randn(num_entities, dim, dtype=torch.float32)
        entity_im = torch.randn(num_entities, dim, dtype=torch.float32)

        return entity_re, entity_im, gamma

    def test_rank_calculation_small(self, small_embeddings):
        """Test rank calculation matches PyTorch reference."""
        entity_re, entity_im, gamma = small_embeddings
        num_entities = entity_re.shape[0]
        dim = entity_re.shape[1]

        validator = TritonDSLFMValidator(
            entity_re=entity_re,
            entity_im=entity_im,
            gamma=gamma,
            device="cuda",
        )

        batch_size = 10
        query_re = torch.randn(batch_size, dim, dtype=torch.float32)
        query_im = torch.randn(batch_size, dim, dtype=torch.float32)
        true_tails = torch.randint(0, num_entities, (batch_size,), dtype=torch.long)

        triton_ranks = validator.compute_ranks(
            query_re.cuda(),
            query_im.cuda(),
            true_tails.cuda(),
        ).cpu()

        pytorch_ranks = _compute_pytorch_ranks(
            query_re.cuda(),
            query_im.cuda(),
            entity_re.cuda(),
            entity_im.cuda(),
            true_tails.cuda(),
            gamma,
        ).cpu()

        assert torch.allclose(triton_ranks.float(), pytorch_ranks.float(), atol=1), (
            f"Rank mismatch: Triton {triton_ranks} vs PyTorch {pytorch_ranks}"
        )

    def test_rank_calculation_medium(self, medium_embeddings):
        """Test rank calculation with more entities."""
        entity_re, entity_im, gamma = medium_embeddings
        num_entities = entity_re.shape[0]
        dim = entity_re.shape[1]

        validator = TritonDSLFMValidator(
            entity_re=entity_re,
            entity_im=entity_im,
            gamma=gamma,
            device="cuda",
        )

        batch_size = 50
        query_re = torch.randn(batch_size, dim, dtype=torch.float32)
        query_im = torch.randn(batch_size, dim, dtype=torch.float32)
        true_tails = torch.randint(0, num_entities, (batch_size,), dtype=torch.long)

        triton_ranks = validator.compute_ranks(
            query_re.cuda(),
            query_im.cuda(),
            true_tails.cuda(),
        ).cpu()

        pytorch_ranks = _compute_pytorch_ranks(
            query_re.cuda(),
            query_im.cuda(),
            entity_re.cuda(),
            entity_im.cuda(),
            true_tails.cuda(),
            gamma,
        ).cpu()

        assert torch.allclose(triton_ranks.float(), pytorch_ranks.float(), atol=1), (
            "Rank mismatch in medium test"
        )

    def test_perfect_rank_for_true_tail(self, small_embeddings):
        """Test that true tail gets rank 1 when it has best score."""
        entity_re, entity_im, gamma = small_embeddings
        num_entities = entity_re.shape[0]

        validator = TritonDSLFMValidator(
            entity_re=entity_re,
            entity_im=entity_im,
            gamma=gamma,
            device="cuda",
        )

        batch_size = 5
        true_tails = torch.randint(0, num_entities, (batch_size,), dtype=torch.long)
        query_re = entity_re[true_tails].clone()
        query_im = entity_im[true_tails].clone()

        triton_ranks = validator.compute_ranks(
            query_re.cuda(),
            query_im.cuda(),
            true_tails.cuda(),
        ).cpu()

        assert (triton_ranks == 1).all(), (
            f"When query == true_tail, rank should be 1, got {triton_ranks}"
        )


def _pc2_forward_reference(pos_probs, parents, root_probs, cond_probs, log_prior):
    batch_size, num_attrs = pos_probs.shape
    neg_probs = 1.0 - pos_probs

    log_prob_y0 = log_prior[0].expand(batch_size)
    log_prob_y1 = log_prior[1].expand(batch_size)

    is_root = parents == -1
    root_probs_expanded = root_probs.unsqueeze(0)

    log_p_attr_given_y0 = pos_probs * torch.log(
        root_probs_expanded[:, :, 0]
    ) + neg_probs * torch.log(1.0 - root_probs_expanded[:, :, 0])
    log_p_attr_given_y1 = pos_probs * torch.log(
        root_probs_expanded[:, :, 1]
    ) + neg_probs * torch.log(1.0 - root_probs_expanded[:, :, 1])

    root_mask = is_root.float().unsqueeze(0)
    log_prob_y0 = log_prob_y0 + (log_p_attr_given_y0 * root_mask).sum(dim=1)
    log_prob_y1 = log_prob_y1 + (log_p_attr_given_y1 * root_mask).sum(dim=1)

    child_mask = (~is_root).float()
    if child_mask.sum() > 0:
        parent_indices = torch.clamp(parents, min=0)
        parent_true = pos_probs.gather(
            1,
            parent_indices.unsqueeze(0).expand(batch_size, -1),
        )

        p0_parent0 = cond_probs[:, 0, 0]
        p0_parent1 = cond_probs[:, 1, 0]
        p1_parent0 = cond_probs[:, 0, 1]
        p1_parent1 = cond_probs[:, 1, 1]

        log_given_parent1_y0 = pos_probs * torch.log(p0_parent1) + neg_probs * torch.log(
            1.0 - p0_parent1
        )
        log_given_parent0_y0 = pos_probs * torch.log(p0_parent0) + neg_probs * torch.log(
            1.0 - p0_parent0
        )
        child_log_y0 = (
            parent_true * log_given_parent1_y0 + (1.0 - parent_true) * log_given_parent0_y0
        )

        log_given_parent1_y1 = pos_probs * torch.log(p1_parent1) + neg_probs * torch.log(
            1.0 - p1_parent1
        )
        log_given_parent0_y1 = pos_probs * torch.log(p1_parent0) + neg_probs * torch.log(
            1.0 - p1_parent0
        )
        child_log_y1 = (
            parent_true * log_given_parent1_y1 + (1.0 - parent_true) * log_given_parent0_y1
        )

        log_prob_y0 = log_prob_y0 + (child_log_y0 * child_mask).sum(dim=1)
        log_prob_y1 = log_prob_y1 + (child_log_y1 * child_mask).sum(dim=1)

    return log_prob_y0, log_prob_y1


class TestTritonPC2Kernels:
    """Tests for PC2 Triton kernels."""

    def test_pc2_forward_matches_reference(self):
        """Execute test pc2 forward matches reference."""

        device = "cuda"
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)

        batch_size = 16
        num_attrs = 32
        pos_probs = torch.rand(batch_size, num_attrs, device=device).clamp(1e-4, 1 - 1e-4)
        parents = torch.randint(
            -1,
            num_attrs,
            (num_attrs,),
            device=device,
            dtype=torch.long,
        )
        root_probs = torch.rand(num_attrs, 2, device=device).clamp(1e-4, 1 - 1e-4)
        cond_probs = torch.rand(num_attrs, 2, 2, device=device).clamp(1e-4, 1 - 1e-4)
        log_prior = torch.log_softmax(torch.randn(2, device=device), dim=0)

        triton_y0, triton_y1 = pc2_forward_triton(
            pos_probs, parents, root_probs, cond_probs, log_prior
        )
        ref_y0, ref_y1 = _pc2_forward_reference(
            pos_probs, parents, root_probs, cond_probs, log_prior
        )

        assert torch.allclose(triton_y0, ref_y0, atol=1e-4, rtol=1e-4)
        assert torch.allclose(triton_y1, ref_y1, atol=1e-4, rtol=1e-4)

    def test_pc2_matrix_forward_matches_reference(self):
        """Execute test pc2 matrix forward matches reference."""

        device = "cuda"
        torch.manual_seed(11)
        torch.cuda.manual_seed_all(11)

        num_heads = 8
        num_tails = 9
        num_attrs = 16
        heads = torch.rand(num_heads, num_attrs, device=device).clamp(1e-4, 1 - 1e-4)
        tails = torch.rand(num_tails, num_attrs, device=device).clamp(1e-4, 1 - 1e-4)
        parents = torch.randint(
            -1,
            num_attrs,
            (num_attrs,),
            device=device,
            dtype=torch.long,
        )
        root_probs = torch.rand(num_attrs, 2, device=device).clamp(1e-4, 1 - 1e-4)
        cond_probs = torch.rand(num_attrs, 2, 2, device=device).clamp(1e-4, 1 - 1e-4)
        log_prior_y1 = float(torch.log_softmax(torch.randn(2, device=device), dim=0)[1].item())

        triton_out = pc2_matrix_forward_triton(
            heads, tails, parents, root_probs, cond_probs, log_prior_y1
        )

        ref = torch.empty((num_heads, num_tails), device=device)
        for h in range(num_heads):
            for t in range(num_tails):
                acc_y1 = log_prior_y1
                for i in range(num_attrs):
                    p_val = 0.5 * (heads[h, i] + tails[t, i])
                    p_neg = 1.0 - p_val
                    parent_idx = parents[i].item()
                    if parent_idx == -1:
                        r_y1 = root_probs[i, 1]
                        acc_y1 += p_val * torch.log(r_y1) + p_neg * torch.log(1.0 - r_y1)
                    else:
                        p_parent = 0.5 * (heads[h, parent_idx] + tails[t, parent_idx])
                        log_p1_y1 = p_val * torch.log(cond_probs[i, 1, 1]) + p_neg * torch.log(
                            1.0 - cond_probs[i, 1, 1]
                        )
                        log_p0_y1 = p_val * torch.log(cond_probs[i, 0, 1]) + p_neg * torch.log(
                            1.0 - cond_probs[i, 0, 1]
                        )
                        acc_y1 += p_parent * log_p1_y1 + (1.0 - p_parent) * log_p0_y1
                ref[h, t] = acc_y1

        assert torch.allclose(triton_out, ref, atol=1e-4, rtol=1e-4)

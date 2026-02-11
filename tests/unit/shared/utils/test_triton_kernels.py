"""
Tests for Triton DSLFM validation kernels.

Verifies correctness by comparing Triton ranks against PyTorch reference.
"""

import pytest
import torch

from pff.shared.acceleration.triton_kernels import (
    TritonDSLFMValidator,
    is_triton_available,
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

        assert torch.allclose(
            triton_ranks.float(), pytorch_ranks.float(), atol=1
        ), f"Rank mismatch: Triton {triton_ranks} vs PyTorch {pytorch_ranks}"

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

        assert torch.allclose(
            triton_ranks.float(), pytorch_ranks.float(), atol=1
        ), "Rank mismatch in medium test"

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

        assert (
            triton_ranks == 1
        ).all(), f"When query == true_tail, rank should be 1, got {triton_ranks}"

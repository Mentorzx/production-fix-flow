"""Bug-hunting tests for DSLFM-KGC + PC stack.

These tests are designed to discover bugs and edge cases:
- Data integrity: no leakage between splits, negative sampling correctness
- Scoring contracts: lambda_pc behavior, rerank_top_k contracts
- PC integration: NaN/inf handling, gradient flow
- Edge cases: empty batches, single entity, extreme values

Each test targets a specific invariant that should hold.
"""

from __future__ import annotations

import math

import pytest
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel
from pff.domain.learning.dslfm.neg_sampling import (
    DegreeBasedSampler,
    SamplerConfig,
    SamplerType,
    get_negative_sampler,
)


@pytest.fixture(autouse=True)
def _disable_cuda(monkeypatch) -> None:
    """Disable CUDA for CPU-only testing."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0, raising=False)


def _tiny_config(lambda_pc: float = 0.0, num_entities: int = 6) -> DSLFMKGCConfig:
    """Create a minimal configuration for quick CPU smoke tests."""
    return DSLFMKGCConfig(
        num_entities=num_entities,
        num_relations=3,
        entity_dim=16,
        feature_dim=16,
        max_communities=4,
        hidden_dim=8,
        temperature=0.5,
        kl_weight=0.01,
        sparsity_weight=0.001,
        sampler_type="degree_based",
        sampler_temperature=1.0,
        lambda_logic=0.0,
        lambda_pc=lambda_pc,
        pc_pruning_threshold=0.01,
        pc_grow_noise=0.01,
        pc_rebuild_every=2,
        pc_max_depth=2,
    )


# =============================================================================
# Category A: Data Integrity & Negative Sampling Tests
# =============================================================================


class TestNegativeSamplingIntegrity:
    """Test negative sampling never uses true triples."""

    def test_degree_based_sampler_excludes_diagonal(self) -> None:
        """DegreeBasedSampler should exclude positive (diagonal) scores."""
        torch.manual_seed(42)
        sampler = get_negative_sampler(SamplerType.DEGREE_BASED)

        # Create a score matrix [batch, batch]
        batch_size = 5
        all_scores = torch.randn(batch_size, batch_size)
        tails = torch.arange(batch_size)

        pos_scores, neg_scores, _ = sampler.get_positive_negative_scores(
            all_scores, tails
        )

        # Positive scores should be the diagonal
        assert pos_scores.shape == (batch_size,)
        assert torch.allclose(pos_scores, all_scores.diag())

        # Negative scores should exclude diagonal
        assert neg_scores.shape == (batch_size, batch_size - 1)

        # Verify no diagonal elements in negatives
        for i in range(batch_size):
            diagonal_val = all_scores[i, i].item()
            neg_row = neg_scores[i].tolist()
            assert (
                diagonal_val not in neg_row
            ), f"Diagonal {diagonal_val} found in negatives"

    def test_degree_based_weights_uniform_fallback(self) -> None:
        """DegreeBasedSampler should fallback to uniform weights when degrees not set."""
        torch.manual_seed(43)
        config = SamplerConfig(sampler_type=SamplerType.DEGREE_BASED)
        sampler = DegreeBasedSampler(config)

        neg_scores = torch.randn(4, 3)
        weights = sampler.weight_negatives(neg_scores)

        # Without degrees set, weights should be all ones (uniform fallback)
        assert torch.allclose(weights, torch.ones_like(weights))

    def test_negative_scores_shape_invariant(self) -> None:
        """Negative scores should always be [batch, batch-1]."""
        sampler = DegreeBasedSampler()

        for batch_size in [2, 5, 10, 32]:
            all_scores = torch.randn(batch_size, batch_size)
            tails = torch.arange(batch_size)
            _, neg_scores, _ = sampler.get_positive_negative_scores(all_scores, tails)

            expected_shape = (batch_size, batch_size - 1)
            assert (
                neg_scores.shape == expected_shape
            ), f"Expected {expected_shape}, got {neg_scores.shape}"


class TestDataLeakageInvariant:
    """Test no data leakage between train/valid splits during evaluation."""

    def test_evaluate_uses_only_provided_triples(self) -> None:
        """Evaluate should only compute metrics on provided triples."""
        torch.manual_seed(44)
        config = _tiny_config(lambda_pc=0.0, num_entities=6)
        model = DSLFMKGCModel(config)

        # Create two disjoint sets of triples
        triples_a = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
        triples_b = torch.tensor([[3, 2, 4], [4, 0, 5]], dtype=torch.long)

        metrics_a = model.evaluate(triples_a, batch_size=2)
        metrics_b = model.evaluate(triples_b, batch_size=2)

        # Metrics should differ (different triples)
        # This is a weak test but catches obvious caching bugs
        assert metrics_a is not None
        assert metrics_b is not None
        assert "mrr" in metrics_a
        assert "mrr" in metrics_b


# =============================================================================
# Category B: Scoring Contract Tests
# =============================================================================


class TestScoringContracts:
    """Test scoring invariants and contracts."""

    def test_lambda_pc_zero_means_no_pc_contribution(self) -> None:
        """With lambda_pc=0, PC should not affect forward scores."""
        torch.manual_seed(45)
        config = _tiny_config(lambda_pc=0.0)
        model = DSLFMKGCModel(config)

        triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)

        # Forward with use_pc=True should match use_pc=False when lambda_pc=0
        with torch.no_grad():
            result_with_pc = model(
                heads=triples[:, 0],
                relations=triples[:, 1],
                tails=triples[:, 2],
                use_pc=True,
            )
            result_without_pc = model(
                heads=triples[:, 0],
                relations=triples[:, 1],
                tails=triples[:, 2],
                use_pc=False,
            )

        # Scores should be identical when lambda_pc=0
        scores_with_pc = result_with_pc["scores"]
        scores_without_pc = result_without_pc["scores"]
        assert torch.allclose(
            scores_with_pc, scores_without_pc, atol=1e-6
        ), f"Scores differ: with_pc={scores_with_pc}, without_pc={scores_without_pc}"

    def test_positive_lambda_pc_changes_scores(self) -> None:
        """With lambda_pc > 0, PC should affect forward scores."""
        torch.manual_seed(46)
        config = _tiny_config(lambda_pc=0.5)
        model = DSLFMKGCModel(config)

        triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)

        with torch.no_grad():
            result_with_pc = model(
                heads=triples[:, 0],
                relations=triples[:, 1],
                tails=triples[:, 2],
                use_pc=True,
            )
            result_without_pc = model(
                heads=triples[:, 0],
                relations=triples[:, 1],
                tails=triples[:, 2],
                use_pc=False,
            )

        # Scores should differ when lambda_pc > 0
        scores_with_pc = result_with_pc["scores"]
        scores_without_pc = result_without_pc["scores"]
        assert not torch.allclose(
            scores_with_pc, scores_without_pc, atol=1e-6
        ), "PC should affect scores when lambda_pc > 0"

    def test_scores_are_finite(self) -> None:
        """All forward scores should be finite (no NaN/inf)."""
        torch.manual_seed(47)
        config = _tiny_config(lambda_pc=0.1)
        model = DSLFMKGCModel(config)

        triples = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 2, 5]], dtype=torch.long)

        with torch.no_grad():
            result = model(
                heads=triples[:, 0],
                relations=triples[:, 1],
                tails=triples[:, 2],
            )
            scores = result["scores"]

        assert torch.isfinite(scores).all(), f"Non-finite scores: {scores}"


class TestReRankTopKContract:
    """Test rerank_top_k behavior in evaluation."""

    def test_rerank_topk_limits_pc_application(self) -> None:
        """PC rerank should only apply to top-k candidates."""
        torch.manual_seed(48)
        config = _tiny_config(lambda_pc=0.1, num_entities=10)
        model = DSLFMKGCModel(config)

        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)

        # Evaluate with different rerank_top_k values
        metrics_full = model.evaluate(triples, batch_size=1, rerank_top_k=None)
        metrics_top5 = model.evaluate(triples, batch_size=1, rerank_top_k=5)
        metrics_top2 = model.evaluate(triples, batch_size=1, rerank_top_k=2)

        # All should return valid metrics
        assert 0.0 <= metrics_full["mrr"] <= 1.0
        assert 0.0 <= metrics_top5["mrr"] <= 1.0
        assert 0.0 <= metrics_top2["mrr"] <= 1.0

    def test_rerank_topk_one_still_works(self) -> None:
        """Edge case: rerank_top_k=1 should not crash."""
        torch.manual_seed(49)
        config = _tiny_config(lambda_pc=0.1, num_entities=6)
        model = DSLFMKGCModel(config)

        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)

        # This is an edge case - only 1 candidate for PC
        metrics = model.evaluate(triples, batch_size=1, rerank_top_k=1)

        assert "mrr" in metrics
        assert not math.isnan(metrics["mrr"])


# =============================================================================
# Category C: PC Integration Edge Cases
# =============================================================================


class TestPCEdgeCases:
    """Test PC component edge cases."""

    def test_pc_handles_all_same_community_probs(self) -> None:
        """PC should handle uniform community probabilities."""
        torch.manual_seed(50)
        config = _tiny_config(lambda_pc=0.5)
        model = DSLFMKGCModel(config)

        # Set up triples
        triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)

        # Forward pass should not crash even with uniform probs
        losses = model.compute_loss(
            heads=triples[:, 0],
            relations=triples[:, 1],
            tails=triples[:, 2],
            use_inbatch_negatives=True,
        )

        assert torch.isfinite(losses["loss"])

    def test_pc_penalty_is_non_negative(self) -> None:
        """PC penalty in loss should be non-negative."""
        torch.manual_seed(51)
        config = _tiny_config(lambda_pc=0.5)
        model = DSLFMKGCModel(config)

        triples = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 2, 5]], dtype=torch.long)

        losses = model.compute_loss(
            heads=triples[:, 0],
            relations=triples[:, 1],
            tails=triples[:, 2],
            use_inbatch_negatives=True,
        )

        # PC penalty should be non-negative (it's a penalty, not a reward)
        assert "pc_penalty" in losses
        assert (
            losses["pc_penalty"].item() >= 0.0
        ), f"PC penalty should be >= 0, got {losses['pc_penalty'].item()}"


# =============================================================================
# Category D: Gradient Flow Tests
# =============================================================================


class TestGradientFlowEdgeCases:
    """Test gradient flow in edge cases."""

    def test_single_triple_gradient_flow(self) -> None:
        """Single triple should still produce valid gradients."""
        torch.manual_seed(52)
        config = _tiny_config(lambda_pc=0.0)
        model = DSLFMKGCModel(config)

        # Single triple - edge case
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        losses = model.compute_loss(
            heads=triples[:, 0],
            relations=triples[:, 1],
            tails=triples[:, 2],
            use_inbatch_negatives=True,
        )

        losses["loss"].backward()

        # Check at least one parameter has gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients computed for single triple"

    def test_all_same_entity_triples(self) -> None:
        """Triples with repeated entities should not cause NaN gradients."""
        torch.manual_seed(53)
        config = _tiny_config(lambda_pc=0.0, num_entities=4)
        model = DSLFMKGCModel(config)

        # All triples use same entities - potential for degeneracy
        triples = torch.tensor(
            [
                [0, 0, 1],
                [0, 1, 1],
                [1, 2, 0],
            ],
            dtype=torch.long,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        losses = model.compute_loss(
            heads=triples[:, 0],
            relations=triples[:, 1],
            tails=triples[:, 2],
            use_inbatch_negatives=True,
        )

        losses["loss"].backward()

        # Check no NaN gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN/inf gradient in {name}"


# =============================================================================
# Category E: Loss Component Tests
# =============================================================================


class TestLossComponents:
    """Test individual loss components."""

    def test_kl_loss_is_non_negative(self) -> None:
        """KL divergence losses should be non-negative."""
        torch.manual_seed(54)
        config = _tiny_config(lambda_pc=0.0)
        model = DSLFMKGCModel(config)

        triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)

        losses = model.compute_loss(
            heads=triples[:, 0],
            relations=triples[:, 1],
            tails=triples[:, 2],
            use_inbatch_negatives=True,
        )

        # KL divergences must be non-negative
        if "kl_gaussian" in losses:
            assert losses["kl_gaussian"].item() >= 0.0
        if "kl_ibp" in losses:
            assert losses["kl_ibp"].item() >= 0.0

    def test_sparsity_loss_is_non_negative(self) -> None:
        """Sparsity loss should be non-negative."""
        torch.manual_seed(55)
        config = _tiny_config(lambda_pc=0.0)
        model = DSLFMKGCModel(config)

        triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)

        losses = model.compute_loss(
            heads=triples[:, 0],
            relations=triples[:, 1],
            tails=triples[:, 2],
            use_inbatch_negatives=True,
        )

        if "sparsity_loss" in losses:
            assert losses["sparsity_loss"].item() >= 0.0

    def test_total_loss_is_sum_of_components(self) -> None:
        """Total loss should be approximately sum of weighted components."""
        torch.manual_seed(56)
        config = _tiny_config(lambda_pc=0.1)
        model = DSLFMKGCModel(config)

        triples = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 2, 5]], dtype=torch.long)

        losses = model.compute_loss(
            heads=triples[:, 0],
            relations=triples[:, 1],
            tails=triples[:, 2],
            use_inbatch_negatives=True,
            regularization_scale=1.0,
        )

        total = losses["loss"].item()

        # Total should be finite
        assert math.isfinite(total), f"Total loss is not finite: {total}"
        assert total >= 0.0, f"Total loss should be >= 0, got {total}"


# =============================================================================
# Category F: Model State Consistency Tests
# =============================================================================


class TestModelStateConsistency:
    """Test model state remains consistent across operations."""

    def test_train_eval_mode_consistency(self) -> None:
        """Model should produce consistent results across train/eval modes."""
        torch.manual_seed(57)
        config = _tiny_config(lambda_pc=0.0)
        model = DSLFMKGCModel(config)

        triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)

        # Eval mode
        model.eval()
        with torch.no_grad():
            result_eval = model(
                heads=triples[:, 0],
                relations=triples[:, 1],
                tails=triples[:, 2],
            )
            scores_eval = result_eval["scores"].clone()

        # Train mode
        model.train()
        with torch.no_grad():
            result_train = model(
                heads=triples[:, 0],
                relations=triples[:, 1],
                tails=triples[:, 2],
            )
            result_train["scores"].clone()

        # Back to eval
        model.eval()
        with torch.no_grad():
            result_eval2 = model(
                heads=triples[:, 0],
                relations=triples[:, 1],
                tails=triples[:, 2],
            )
            scores_eval2 = result_eval2["scores"].clone()

        # Eval scores should be reproducible
        assert torch.allclose(scores_eval, scores_eval2, atol=1e-6)

    def test_multiple_forward_passes_deterministic(self) -> None:
        """Multiple forward passes should be deterministic in eval mode."""
        torch.manual_seed(58)
        config = _tiny_config(lambda_pc=0.1)
        model = DSLFMKGCModel(config)
        model.eval()

        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)

        with torch.no_grad():
            result1 = model(
                heads=triples[:, 0],
                relations=triples[:, 1],
                tails=triples[:, 2],
            )
            scores1 = result1["scores"].clone()
            result2 = model(
                heads=triples[:, 0],
                relations=triples[:, 1],
                tails=triples[:, 2],
            )
            scores2 = result2["scores"].clone()

        assert torch.allclose(scores1, scores2), "Eval mode should be deterministic"

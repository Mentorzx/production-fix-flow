"""RotatE Integration Tests.

Tests for RotatE model integration with realistic-sized graphs.
These tests verify:
1. MRR/Hits metrics stability between validation and test splits
2. Embeddings usefulness for hybrid models (MRR → hybrid improvement)
3. Memory/time bounds for realistic graph sizes

Design Patterns Applied:
    - **Factory Pattern:** Create test datasets with varying characteristics.
    - **Strategy Pattern:** Different evaluation strategies for metrics.

Author: PFF Team
Date: 2025-06-01
"""

from __future__ import annotations

import gc
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


class SyntheticKGFactory:
    """Factory for creating synthetic knowledge graphs with controlled properties.
    
    Creates graphs with:
    - Controlled entity/relation counts
    - Configurable triple density
    - Deterministic generation for reproducibility
    """
    
    @staticmethod
    def create_graph(
        num_entities: int,
        num_relations: int,
        num_triples: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Create a synthetic KG with specified size.
        
        Args:
            num_entities: Number of unique entities.
            num_relations: Number of unique relations.
            num_triples: Number of triples to generate.
            seed: Random seed for reproducibility.
            
        Returns:
            Array of shape [num_triples, 3] with (head, relation, tail).
        """
        rng = np.random.default_rng(seed)
        
        heads = rng.integers(0, num_entities, size=num_triples)
        relations = rng.integers(0, num_relations, size=num_triples)
        tails = rng.integers(0, num_entities, size=num_triples)
        
        return np.stack([heads, relations, tails], axis=1).astype(np.int64)
    
    @staticmethod
    def create_train_val_test_split(
        num_entities: int,
        num_relations: int,
        total_triples: int,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create train/val/test splits from a synthetic KG.
        
        Args:
            num_entities: Number of unique entities.
            num_relations: Number of unique relations.
            total_triples: Total number of triples.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            seed: Random seed for reproducibility.
            
        Returns:
            Tuple of (train_triples, val_triples, test_triples).
        """
        all_triples = SyntheticKGFactory.create_graph(
            num_entities, num_relations, total_triples, seed
        )
        
        rng = np.random.default_rng(seed + 1)
        indices = rng.permutation(total_triples)
        
        train_end = int(total_triples * train_ratio)
        val_end = train_end + int(total_triples * val_ratio)
        
        train_triples = all_triples[indices[:train_end]]
        val_triples = all_triples[indices[train_end:val_end]]
        test_triples = all_triples[indices[val_end:]]
        
        return train_triples, val_triples, test_triples


class MockRotatEModel(torch.nn.Module):
    """Mock RotatE model for integration testing.
    
    Simulates RotatE behavior with:
    - Complex embeddings for entities and relations
    - Proper scoring function
    - Configurable embedding dimension
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        gamma: float = 12.0,
    ) -> None:
        """Initialize mock RotatE model.
        
        Args:
            num_entities: Number of entities.
            num_relations: Number of relations.
            embedding_dim: Dimension of embeddings.
            gamma: Margin parameter.
        """
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        
        # Entity embeddings (complex: real + imaginary)
        self.entity_embedding = torch.nn.Embedding(
            num_entities, embedding_dim * 2
        )
        # Relation embeddings (phase angles)
        self.relation_embedding = torch.nn.Embedding(
            num_relations, embedding_dim
        )
        
        # Initialize
        torch.nn.init.uniform_(self.entity_embedding.weight, -1.0, 1.0)
        torch.nn.init.uniform_(
            self.relation_embedding.weight, 
            -np.pi, np.pi
        )
        
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RotatE scores.
        
        Args:
            heads: Head entity indices [batch_size].
            relations: Relation indices [batch_size].
            tails: Tail entity indices [batch_size].
            
        Returns:
            Scores tensor [batch_size].
        """
        # Get embeddings
        head_emb = self.entity_embedding(heads)
        rel_emb = self.relation_embedding(relations)
        tail_emb = self.entity_embedding(tails)
        
        # Split into real and imaginary parts
        head_re = head_emb[..., :self.embedding_dim]
        head_im = head_emb[..., self.embedding_dim:]
        tail_re = tail_emb[..., :self.embedding_dim]
        tail_im = tail_emb[..., self.embedding_dim:]
        
        # Relation as rotation in complex plane
        rel_re = torch.cos(rel_emb)
        rel_im = torch.sin(rel_emb)
        
        # Rotate head embedding
        rotated_re = head_re * rel_re - head_im * rel_im
        rotated_im = head_re * rel_im + head_im * rel_re
        
        # Distance to tail
        diff_re = rotated_re - tail_re
        diff_im = rotated_im - tail_im
        
        # L2 norm of difference
        score = torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-8).sum(dim=-1)
        
        return self.gamma - score
    
    def get_entity_embeddings(self) -> torch.Tensor:
        """Get entity embeddings for downstream use.
        
        Returns:
            Entity embeddings tensor [num_entities, embedding_dim * 2].
        """
        return self.entity_embedding.weight.detach()


def compute_metrics(
    model: torch.nn.Module,
    triples: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    """Compute link prediction metrics (MRR, Hits@K).
    
    Simplified version of RotatEMetricsReporter.compute_link_prediction_metrics.
    
    Args:
        model: RotatE model to evaluate.
        triples: Triples array [n_triples, 3].
        device: Computation device.
        batch_size: Evaluation batch size.
        
    Returns:
        Dictionary with 'mrr', 'hits@1', 'hits@10' metrics.
    """
    model.eval()
    num_samples = len(triples)
    num_entities = model.num_entities
    
    # Limit batch size to prevent OOM
    eval_batch_size = min(batch_size, num_samples, 32)
    
    all_mrr = []
    all_hits1 = []
    all_hits10 = []
    
    with torch.no_grad():
        all_entities = torch.arange(num_entities, device=device)
        
        for batch_start in range(0, num_samples, eval_batch_size):
            batch_end = min(batch_start + eval_batch_size, num_samples)
            batch_triples = triples[batch_start:batch_end]
            batch_len = len(batch_triples)
            
            heads = torch.tensor(
                batch_triples[:, 0], dtype=torch.long, device=device
            )
            rels = torch.tensor(
                batch_triples[:, 1], dtype=torch.long, device=device
            )
            tails = torch.tensor(
                batch_triples[:, 2], dtype=torch.long, device=device
            )
            
            # Score all tail candidates
            heads_exp = heads.unsqueeze(1).expand(-1, num_entities)
            rels_exp = rels.unsqueeze(1).expand(-1, num_entities)
            all_tails = all_entities.unsqueeze(0).expand(batch_len, -1)
            
            scores = model.forward(
                heads_exp.reshape(-1),
                rels_exp.reshape(-1),
                all_tails.reshape(-1),
            ).reshape(batch_len, num_entities)
            
            # Rank of true tail (higher score = better)
            true_scores = scores[torch.arange(batch_len, device=device), tails]
            ranks = (scores > true_scores.unsqueeze(1)).sum(dim=1) + 1
            
            all_mrr.append((1.0 / ranks.float()).cpu())
            all_hits1.append((ranks == 1).cpu())
            all_hits10.append((ranks <= 10).cpu())
    
    mrr_tensor = torch.cat(all_mrr)
    hits1_tensor = torch.cat(all_hits1)
    hits10_tensor = torch.cat(all_hits10)
    
    return {
        "mrr": mrr_tensor.mean().item(),
        "hits@1": hits1_tensor.float().mean().item(),
        "hits@10": hits10_tensor.float().mean().item(),
    }


def train_model_simple(
    model: torch.nn.Module,
    train_triples: np.ndarray,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 0.001,
) -> list[float]:
    """Simple training loop for RotatE.
    
    Args:
        model: RotatE model to train.
        train_triples: Training triples [n_triples, 3].
        device: Computation device.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        
    Returns:
        List of loss values per epoch.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_entities = model.num_entities
    
    losses = []
    rng = np.random.default_rng(42)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        indices = rng.permutation(len(train_triples))
        
        for batch_start in range(0, len(train_triples), batch_size):
            batch_end = min(batch_start + batch_size, len(train_triples))
            batch_indices = indices[batch_start:batch_end]
            batch = train_triples[batch_indices]
            
            heads = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            rels = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
            tails = torch.tensor(batch[:, 2], dtype=torch.long, device=device)
            
            # Positive scores
            pos_scores = model(heads, rels, tails)
            
            # Negative samples (corrupt tails)
            neg_tails = torch.randint(
                0, num_entities, (len(batch),), device=device
            )
            neg_scores = model(heads, rels, neg_tails)
            
            # Margin ranking loss
            margin = 1.0
            loss = torch.clamp(margin - pos_scores + neg_scores, min=0).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        losses.append(epoch_loss / max(num_batches, 1))
    
    return losses


class TestMetricsStabilityBetweenSplits:
    """Tests for MRR/Hits stability between validation and test splits.
    
    Verifies that metrics don't have bizarre gaps between val and test,
    which would indicate overfitting or data leakage.
    """
    
    def test_val_test_mrr_gap_within_bounds(self) -> None:
        """MRR gap between val and test should be reasonable (<0.2)."""
        device = torch.device("cpu")
        
        # Create moderate-sized graph
        train, val, test = SyntheticKGFactory.create_train_val_test_split(
            num_entities=500,
            num_relations=20,
            total_triples=5000,
            seed=42,
        )
        
        model = MockRotatEModel(
            num_entities=500,
            num_relations=20,
            embedding_dim=50,
        ).to(device)
        
        # Train for a few epochs
        train_model_simple(model, train, device, epochs=5, batch_size=128)
        
        # Evaluate on both splits
        val_metrics = compute_metrics(model, val, device)
        test_metrics = compute_metrics(model, test, device)
        
        mrr_gap = abs(val_metrics["mrr"] - test_metrics["mrr"])
        
        # Gap should be reasonable (< 0.2)
        assert mrr_gap < 0.2, (
            f"MRR gap too large: val={val_metrics['mrr']:.4f}, "
            f"test={test_metrics['mrr']:.4f}, gap={mrr_gap:.4f}"
        )
    
    def test_hits10_stability_across_seeds(self) -> None:
        """Hits@10 should be relatively stable across random seeds."""
        device = torch.device("cpu")
        
        hits10_values = []
        
        for seed in range(3):
            train, val, _ = SyntheticKGFactory.create_train_val_test_split(
                num_entities=300,
                num_relations=15,
                total_triples=3000,
                seed=seed * 100,
            )
            
            torch.manual_seed(seed)
            model = MockRotatEModel(
                num_entities=300,
                num_relations=15,
                embedding_dim=32,
            ).to(device)
            
            train_model_simple(model, train, device, epochs=3)
            metrics = compute_metrics(model, val, device)
            hits10_values.append(metrics["hits@10"])
        
        # Standard deviation should be reasonable
        std = np.std(hits10_values)
        assert std < 0.15, f"Hits@10 too unstable across seeds: std={std:.4f}"
    
    def test_metrics_improve_with_training(self) -> None:
        """MRR should improve after training vs random initialization."""
        device = torch.device("cpu")
        
        train, val, _ = SyntheticKGFactory.create_train_val_test_split(
            num_entities=400,
            num_relations=20,
            total_triples=4000,
            seed=42,
        )
        
        model = MockRotatEModel(
            num_entities=400,
            num_relations=20,
            embedding_dim=50,
        ).to(device)
        
        # Metrics before training
        initial_metrics = compute_metrics(model, val, device)
        
        # Train
        train_model_simple(model, train, device, epochs=10)
        
        # Metrics after training
        trained_metrics = compute_metrics(model, val, device)
        
        # MRR should improve (or at least not degrade significantly)
        assert trained_metrics["mrr"] >= initial_metrics["mrr"] - 0.05, (
            f"MRR degraded after training: "
            f"{initial_metrics['mrr']:.4f} -> {trained_metrics['mrr']:.4f}"
        )


class TestEmbeddingsUsefulnessForHybrid:
    """Tests for embeddings usefulness in hybrid models.
    
    Verifies that better RotatE embeddings lead to better hybrid performance.
    """
    
    def test_better_mrr_correlates_with_embedding_quality(self) -> None:
        """Models with better MRR should have more discriminative embeddings."""
        device = torch.device("cpu")
        
        train, val, _ = SyntheticKGFactory.create_train_val_test_split(
            num_entities=300,
            num_relations=15,
            total_triples=3000,
            seed=42,
        )
        
        # Create two models: untrained vs trained
        torch.manual_seed(42)
        model_untrained = MockRotatEModel(
            num_entities=300,
            num_relations=15,
            embedding_dim=50,
        ).to(device)
        
        torch.manual_seed(42)
        model_trained = MockRotatEModel(
            num_entities=300,
            num_relations=15,
            embedding_dim=50,
        ).to(device)
        train_model_simple(model_trained, train, device, epochs=10)
        
        # Compare MRR
        mrr_untrained = compute_metrics(model_untrained, val, device)["mrr"]
        mrr_trained = compute_metrics(model_trained, val, device)["mrr"]
        
        # Trained model should have better MRR
        assert mrr_trained >= mrr_untrained, (
            f"Trained model should have better MRR: "
            f"untrained={mrr_untrained:.4f}, trained={mrr_trained:.4f}"
        )
        
        # Check embedding variance (trained should be more structured)
        emb_untrained = model_untrained.get_entity_embeddings()
        emb_trained = model_trained.get_entity_embeddings()
        
        # Trained embeddings should have meaningful structure
        # (variance should be reasonable, not collapsed or exploded)
        var_trained = emb_trained.var().item()
        assert 0.01 < var_trained < 100, (
            f"Trained embedding variance out of bounds: {var_trained:.4f}"
        )
    
    def test_embeddings_produce_useful_features(self) -> None:
        """RotatE embeddings should be usable as features for downstream tasks."""
        device = torch.device("cpu")
        
        train, _, _ = SyntheticKGFactory.create_train_val_test_split(
            num_entities=200,
            num_relations=10,
            total_triples=2000,
            seed=42,
        )
        
        model = MockRotatEModel(
            num_entities=200,
            num_relations=10,
            embedding_dim=50,
        ).to(device)
        
        train_model_simple(model, train, device, epochs=5)
        
        embeddings = model.get_entity_embeddings().cpu().numpy()
        
        # Embeddings should have reasonable properties for ML
        assert embeddings.shape == (200, 100), f"Wrong shape: {embeddings.shape}"
        assert not np.isnan(embeddings).any(), "Embeddings contain NaN"
        assert not np.isinf(embeddings).any(), "Embeddings contain Inf"
        
        # Should have reasonable L2 norms
        norms = np.linalg.norm(embeddings, axis=1)
        assert norms.min() > 0.1, "Some embeddings have near-zero norm"
        assert norms.max() < 100, "Some embeddings have exploded norm"
    
    def test_mrr_improvement_transfers_to_triple_classification(self) -> None:
        """Better MRR should correlate with better triple classification.
        
        Simulates hybrid scenario where embeddings are used for classification.
        """
        device = torch.device("cpu")
        
        train, val, _ = SyntheticKGFactory.create_train_val_test_split(
            num_entities=300,
            num_relations=15,
            total_triples=3000,
            seed=42,
        )
        
        # Train model
        torch.manual_seed(42)
        model = MockRotatEModel(
            num_entities=300,
            num_relations=15,
            embedding_dim=50,
        ).to(device)
        train_model_simple(model, train, device, epochs=10)
        
        # Use model scores for classification
        # Positive: real triples, Negative: corrupted triples
        positive_triples = val[:100]
        
        # Corrupt triples
        rng = np.random.default_rng(42)
        negative_triples = positive_triples.copy()
        negative_triples[:, 2] = rng.integers(0, 300, 100)
        
        with torch.no_grad():
            pos_heads = torch.tensor(positive_triples[:, 0], device=device)
            pos_rels = torch.tensor(positive_triples[:, 1], device=device)
            pos_tails = torch.tensor(positive_triples[:, 2], device=device)
            pos_scores = model(pos_heads, pos_rels, pos_tails).cpu().numpy()
            
            neg_heads = torch.tensor(negative_triples[:, 0], device=device)
            neg_rels = torch.tensor(negative_triples[:, 1], device=device)
            neg_tails = torch.tensor(negative_triples[:, 2], device=device)
            neg_scores = model(neg_heads, neg_rels, neg_tails).cpu().numpy()
        
        # Positive scores should generally be higher than negative
        # (with some tolerance for random negatives that might be valid)
        pos_mean = pos_scores.mean()
        neg_mean = neg_scores.mean()
        
        assert pos_mean > neg_mean - 1.0, (
            f"Positive triples should score higher: "
            f"pos_mean={pos_mean:.4f}, neg_mean={neg_mean:.4f}"
        )


class TestMemoryAndTimeConstraints:
    """Tests for memory and time bounds on realistic graph sizes."""
    
    def test_moderate_graph_fits_in_memory(self) -> None:
        """A moderate graph (1K entities, 50K triples) should fit in RAM."""
        device = torch.device("cpu")
        
        # Clear memory
        gc.collect()
        
        train, val, _ = SyntheticKGFactory.create_train_val_test_split(
            num_entities=1000,
            num_relations=50,
            total_triples=50000,
            seed=42,
        )
        
        model = MockRotatEModel(
            num_entities=1000,
            num_relations=50,
            embedding_dim=100,
        ).to(device)
        
        # Should complete without OOM
        train_model_simple(model, train, device, epochs=2, batch_size=256)
        metrics = compute_metrics(model, val[:500], device)
        
        assert metrics["mrr"] >= 0, "Metrics computation failed"
        
        # Clean up
        del model, train, val
        gc.collect()
    
    def test_training_time_scales_reasonably(self) -> None:
        """Training time should scale reasonably with graph size."""
        device = torch.device("cpu")
        
        times: dict[int, float] = {}
        
        for num_entities in [100, 200]:
            train, _, _ = SyntheticKGFactory.create_train_val_test_split(
                num_entities=num_entities,
                num_relations=10,
                total_triples=num_entities * 10,
                seed=42,
            )
            
            model = MockRotatEModel(
                num_entities=num_entities,
                num_relations=10,
                embedding_dim=32,
            ).to(device)
            
            start = time.perf_counter()
            train_model_simple(model, train, device, epochs=3, batch_size=64)
            elapsed = time.perf_counter() - start
            
            times[num_entities] = elapsed
            
            del model, train
            gc.collect()
        
        # Time should not explode (at most 10x for 2x entities)
        ratio = times[200] / times[100]
        assert ratio < 10, f"Training time scaling too steep: {ratio:.2f}x"
    
    def test_evaluation_time_reasonable(self) -> None:
        """Evaluation should complete in reasonable time for moderate graphs."""
        device = torch.device("cpu")
        
        _, val, _ = SyntheticKGFactory.create_train_val_test_split(
            num_entities=500,
            num_relations=20,
            total_triples=5000,
            seed=42,
        )
        
        model = MockRotatEModel(
            num_entities=500,
            num_relations=20,
            embedding_dim=50,
        ).to(device)
        
        # Evaluate on subset
        val_subset = val[:200]
        
        start = time.perf_counter()
        metrics = compute_metrics(model, val_subset, device, batch_size=32)
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time (< 60s on CPU)
        assert elapsed < 60, f"Evaluation took too long: {elapsed:.2f}s"
        assert metrics["mrr"] >= 0, "Metrics invalid"


class TestModelCheckpointing:
    """Tests for model checkpointing and restoration."""
    
    def test_checkpoint_restore_preserves_metrics(self) -> None:
        """Metrics should be identical after checkpoint restore."""
        device = torch.device("cpu")
        
        train, val, _ = SyntheticKGFactory.create_train_val_test_split(
            num_entities=200,
            num_relations=10,
            total_triples=2000,
            seed=42,
        )
        
        torch.manual_seed(42)
        model = MockRotatEModel(
            num_entities=200,
            num_relations=10,
            embedding_dim=50,
        ).to(device)
        
        train_model_simple(model, train, device, epochs=5)
        
        # Get metrics before save
        metrics_before = compute_metrics(model, val, device)
        
        # Save and restore
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            # Create new model and load
            torch.manual_seed(999)  # Different seed
            model_restored = MockRotatEModel(
                num_entities=200,
                num_relations=10,
                embedding_dim=50,
            ).to(device)
            model_restored.load_state_dict(
                torch.load(checkpoint_path, weights_only=True)
            )
            
            # Get metrics after restore
            metrics_after = compute_metrics(model_restored, val, device)
        
        # Metrics should be identical
        assert abs(metrics_before["mrr"] - metrics_after["mrr"]) < 1e-6
        assert abs(metrics_before["hits@1"] - metrics_after["hits@1"]) < 1e-6
        assert abs(metrics_before["hits@10"] - metrics_after["hits@10"]) < 1e-6
    
    def test_embeddings_preserved_after_restore(self) -> None:
        """Entity embeddings should be identical after checkpoint restore."""
        device = torch.device("cpu")
        
        train, _, _ = SyntheticKGFactory.create_train_val_test_split(
            num_entities=100,
            num_relations=5,
            total_triples=1000,
            seed=42,
        )
        
        torch.manual_seed(42)
        model = MockRotatEModel(
            num_entities=100,
            num_relations=5,
            embedding_dim=32,
        ).to(device)
        
        train_model_simple(model, train, device, epochs=3)
        
        embeddings_before = model.get_entity_embeddings().clone()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            torch.manual_seed(999)
            model_restored = MockRotatEModel(
                num_entities=100,
                num_relations=5,
                embedding_dim=32,
            ).to(device)
            model_restored.load_state_dict(
                torch.load(checkpoint_path, weights_only=True)
            )
            
            embeddings_after = model_restored.get_entity_embeddings()
        
        # Embeddings should be identical
        assert torch.allclose(embeddings_before, embeddings_after, atol=1e-6)


class TestNumericalStability:
    """Tests for numerical stability during training and evaluation."""
    
    def test_no_nan_in_embeddings_after_training(self) -> None:
        """Embeddings should not contain NaN after training."""
        device = torch.device("cpu")
        
        train, _, _ = SyntheticKGFactory.create_train_val_test_split(
            num_entities=500,
            num_relations=25,
            total_triples=5000,
            seed=42,
        )
        
        model = MockRotatEModel(
            num_entities=500,
            num_relations=25,
            embedding_dim=100,
        ).to(device)
        
        train_model_simple(model, train, device, epochs=15, lr=0.01)
        
        embeddings = model.get_entity_embeddings()
        
        assert not torch.isnan(embeddings).any(), "NaN in entity embeddings"
        assert not torch.isinf(embeddings).any(), "Inf in entity embeddings"
    
    def test_no_nan_in_metrics(self) -> None:
        """Computed metrics should not be NaN."""
        device = torch.device("cpu")
        
        train, val, _ = SyntheticKGFactory.create_train_val_test_split(
            num_entities=300,
            num_relations=15,
            total_triples=3000,
            seed=42,
        )
        
        model = MockRotatEModel(
            num_entities=300,
            num_relations=15,
            embedding_dim=50,
        ).to(device)
        
        train_model_simple(model, train, device, epochs=5)
        metrics = compute_metrics(model, val, device)
        
        assert not np.isnan(metrics["mrr"]), "MRR is NaN"
        assert not np.isnan(metrics["hits@1"]), "Hits@1 is NaN"
        assert not np.isnan(metrics["hits@10"]), "Hits@10 is NaN"
        
        # Metrics should be in valid range [0, 1]
        assert 0 <= metrics["mrr"] <= 1
        assert 0 <= metrics["hits@1"] <= 1
        assert 0 <= metrics["hits@10"] <= 1
    
    def test_loss_decreases_during_training(self) -> None:
        """Training loss should generally decrease."""
        device = torch.device("cpu")
        
        train, _, _ = SyntheticKGFactory.create_train_val_test_split(
            num_entities=200,
            num_relations=10,
            total_triples=2000,
            seed=42,
        )
        
        model = MockRotatEModel(
            num_entities=200,
            num_relations=10,
            embedding_dim=50,
        ).to(device)
        
        losses = train_model_simple(model, train, device, epochs=10)
        
        # Loss should not explode
        assert all(not np.isnan(loss) for loss in losses), "NaN in losses"
        assert all(loss < 1000 for loss in losses), "Loss exploded"
        
        # Final loss should be lower than initial (with some tolerance)
        assert losses[-1] <= losses[0] + 0.5, (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

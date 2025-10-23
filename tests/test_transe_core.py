"""
Tests for TransE Core Implementation (pff/validators/transe/core.py)

Sprint 6: Testes IA/ML - TransE

Tests cover:
- Model initialization
- Forward pass computation
- Embedding initialization (Xavier uniform)
- Score triple functionality
- Embedding normalization
- Loss calculation
- Negative sampling
- Distance metrics (L1 and L2 norms)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from pff.validators.transe.core import TransEModel, TransEDataset


@pytest.fixture
def small_kg_params():
    """Small knowledge graph parameters for testing."""
    return {
        "num_entities": 100,
        "num_relations": 10,
        "embedding_dim": 32,
        "margin": 1.0,
        "norm": 2,  # L2 norm
    }


@pytest.fixture
def transe_model(small_kg_params):
    """Create a small TransE model for testing."""
    return TransEModel(**small_kg_params)


@pytest.fixture
def sample_triples():
    """Sample triples for testing (h, r, t)."""
    return np.array([
        [0, 0, 1],   # Entity 0 -[relation 0]-> Entity 1
        [1, 1, 2],   # Entity 1 -[relation 1]-> Entity 2
        [2, 2, 3],   # Entity 2 -[relation 2]-> Entity 3
        [3, 0, 4],   # Entity 3 -[relation 0]-> Entity 4
        [4, 1, 5],   # Entity 4 -[relation 1]-> Entity 5
    ])


# ═══════════════════════════════════════════════════════════════════
# Model Initialization Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTransEModelInit:
    """Test TransE model initialization."""

    def test_model_initialization(self, small_kg_params):
        """Test model initializes with correct parameters."""
        model = TransEModel(**small_kg_params)

        assert model.num_entities == 100
        assert model.num_relations == 10
        assert model.embedding_dim == 32
        assert model.margin == 1.0
        assert model.norm == 2

    def test_embedding_layers_created(self, transe_model):
        """Test entity and relation embedding layers are created."""
        assert isinstance(transe_model.entity_embeddings, nn.Embedding)
        assert isinstance(transe_model.relation_embeddings, nn.Embedding)

        # Check embedding shapes
        assert transe_model.entity_embeddings.weight.shape == (100, 32)
        assert transe_model.relation_embeddings.weight.shape == (10, 32)

    def test_embedding_initialization_range(self, transe_model):
        """Test embeddings are initialized in reasonable range (Xavier)."""
        entity_weights = transe_model.entity_embeddings.weight.data
        relation_weights = transe_model.relation_embeddings.weight.data

        # Xavier uniform typically initializes in range [-sqrt(6/(in+out)), sqrt(6/(in+out))]
        # For our case: sqrt(6/(1+32)) ≈ 0.43
        assert entity_weights.abs().max() < 1.0  # Reasonable range
        assert relation_weights.abs().max() < 1.0

    def test_entity_embeddings_normalized(self, transe_model):
        """Test entity embeddings are normalized to unit length."""
        entity_weights = transe_model.entity_embeddings.weight.data

        # Calculate L2 norms of each embedding
        norms = torch.norm(entity_weights, p=2, dim=1)

        # All norms should be close to 1.0 (normalized)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_model_with_l1_norm(self):
        """Test model can be initialized with L1 norm."""
        model = TransEModel(
            num_entities=50,
            num_relations=5,
            embedding_dim=16,
            margin=1.0,
            norm=1,  # L1 norm
        )

        assert model.norm == 1


# ═══════════════════════════════════════════════════════════════════
# Forward Pass Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTransEForward:
    """Test TransE forward pass."""

    def test_forward_pass_shape(self, transe_model):
        """Test forward pass returns correct shape."""
        batch_size = 16
        heads = torch.randint(0, 100, (batch_size,))
        relations = torch.randint(0, 10, (batch_size,))
        tails = torch.randint(0, 100, (batch_size,))

        scores = transe_model.forward(heads, relations, tails)

        assert scores.shape == (batch_size,)

    def test_forward_pass_computation(self, transe_model):
        """Test forward pass computes scores correctly."""
        heads = torch.tensor([0, 1, 2])
        relations = torch.tensor([0, 1, 0])
        tails = torch.tensor([1, 2, 3])

        scores = transe_model.forward(heads, relations, tails)

        # Scores should be negative distances (higher is better)
        assert scores.shape == (3,)
        assert all(scores <= 0)  # All scores should be negative or zero

    def test_forward_pass_deterministic(self, transe_model):
        """Test forward pass is deterministic."""
        heads = torch.tensor([0, 1])
        relations = torch.tensor([0, 1])
        tails = torch.tensor([1, 2])

        scores1 = transe_model.forward(heads, relations, tails)
        scores2 = transe_model.forward(heads, relations, tails)

        assert torch.allclose(scores1, scores2)

    def test_forward_clamping_out_of_bounds(self, transe_model):
        """Test forward pass clamps out-of-bounds indices."""
        # Out of bounds indices
        heads = torch.tensor([0, 999])  # 999 is out of bounds
        relations = torch.tensor([0, 0])
        tails = torch.tensor([1, 1])

        # Should not raise error due to clamping
        scores = transe_model.forward(heads, relations, tails)
        assert scores.shape == (2,)

    def test_forward_translational_property(self, transe_model):
        """Test that h + r ≈ t has lower distance (higher score)."""
        # Get embeddings for a specific triple
        head_idx = torch.tensor([0])
        rel_idx = torch.tensor([0])
        tail_idx = torch.tensor([1])

        # Compute score for correct triple
        correct_score = transe_model.forward(head_idx, rel_idx, tail_idx)

        # Compute score for incorrect triple (wrong tail)
        wrong_tail = torch.tensor([50])
        wrong_score = transe_model.forward(head_idx, rel_idx, wrong_tail)

        # After training, correct triples should have higher scores
        # Here we just test that scores are computable
        assert isinstance(correct_score.item(), float)
        assert isinstance(wrong_score.item(), float)


# ═══════════════════════════════════════════════════════════════════
# Score Triple Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTransEScoring:
    """Test TransE triple scoring."""

    def test_score_triple_single(self, transe_model):
        """Test scoring a single triple."""
        score = transe_model.score_triple(head_idx=0, rel_idx=0, tail_idx=1)

        assert isinstance(score, float)
        assert score <= 0  # Negative distance

    def test_score_triple_deterministic(self, transe_model):
        """Test score_triple is deterministic."""
        score1 = transe_model.score_triple(0, 0, 1)
        score2 = transe_model.score_triple(0, 0, 1)

        assert score1 == score2

    def test_score_triple_different_triples(self, transe_model):
        """Test different triples get different scores."""
        score1 = transe_model.score_triple(0, 0, 1)
        score2 = transe_model.score_triple(2, 1, 3)

        # Different triples should (very likely) have different scores
        assert score1 != score2


# ═══════════════════════════════════════════════════════════════════
# Embedding Normalization Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTransENormalization:
    """Test TransE embedding normalization."""

    def test_normalize_embeddings(self, transe_model):
        """Test embedding normalization."""
        # Manually corrupt entity embeddings
        with torch.no_grad():
            transe_model.entity_embeddings.weight.data *= 5.0  # Scale up

        # Verify embeddings are not normalized
        norms_before = torch.norm(transe_model.entity_embeddings.weight.data, p=2, dim=1)
        assert not torch.allclose(norms_before, torch.ones_like(norms_before), atol=1e-6)

        # Normalize
        transe_model.normalize_embeddings()

        # Verify embeddings are now normalized
        norms_after = torch.norm(transe_model.entity_embeddings.weight.data, p=2, dim=1)
        assert torch.allclose(norms_after, torch.ones_like(norms_after), atol=1e-6)

    def test_normalization_preserves_direction(self, transe_model):
        """Test normalization preserves embedding direction."""
        # Get original embeddings
        original_emb = transe_model.entity_embeddings.weight.data[0].clone()
        original_direction = original_emb / torch.norm(original_emb)

        # Scale and normalize
        with torch.no_grad():
            transe_model.entity_embeddings.weight.data[0] *= 3.0

        transe_model.normalize_embeddings()

        # New embedding should point in same direction
        new_emb = transe_model.entity_embeddings.weight.data[0]
        assert torch.allclose(new_emb, original_direction, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
# Distance Metrics Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTransEDistanceMetrics:
    """Test TransE distance metrics (L1 vs L2)."""

    def test_l2_norm_distance(self):
        """Test L2 norm distance calculation."""
        model = TransEModel(
            num_entities=10,
            num_relations=2,
            embedding_dim=8,
            norm=2,  # L2
        )

        heads = torch.tensor([0])
        relations = torch.tensor([0])
        tails = torch.tensor([1])

        score = model.forward(heads, relations, tails)

        # Score should be negative (negative distance)
        assert score.item() <= 0

    def test_l1_norm_distance(self):
        """Test L1 norm distance calculation."""
        model = TransEModel(
            num_entities=10,
            num_relations=2,
            embedding_dim=8,
            norm=1,  # L1
        )

        heads = torch.tensor([0])
        relations = torch.tensor([0])
        tails = torch.tensor([1])

        score = model.forward(heads, relations, tails)

        # Score should be negative (negative distance)
        assert score.item() <= 0

    def test_l1_vs_l2_distances_differ(self):
        """Test L1 and L2 norms produce different scores."""
        # Create identical models with different norms
        model_l1 = TransEModel(num_entities=10, num_relations=2, embedding_dim=8, norm=1)
        model_l2 = TransEModel(num_entities=10, num_relations=2, embedding_dim=8, norm=2)

        # Copy embeddings to make them identical
        with torch.no_grad():
            model_l2.entity_embeddings.weight.data = model_l1.entity_embeddings.weight.data.clone()
            model_l2.relation_embeddings.weight.data = model_l1.relation_embeddings.weight.data.clone()

        heads = torch.tensor([0, 1])
        relations = torch.tensor([0, 1])
        tails = torch.tensor([1, 2])

        score_l1 = model_l1.forward(heads, relations, tails)
        score_l2 = model_l2.forward(heads, relations, tails)

        # L1 and L2 distances should be different
        assert not torch.allclose(score_l1, score_l2)


# ═══════════════════════════════════════════════════════════════════
# TransEDataset Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTransEDataset:
    """Test TransEDataset functionality."""

    def test_dataset_initialization(self, sample_triples):
        """Test dataset initializes correctly."""
        dataset = TransEDataset(
            triples=sample_triples,
            num_entities=100,
            num_negatives=1,
        )

        assert len(dataset) == len(sample_triples)
        assert dataset.num_entities == 100
        assert dataset.num_negatives == 1

    def test_dataset_getitem_shape(self, sample_triples):
        """Test __getitem__ returns correct shapes."""
        dataset = TransEDataset(
            triples=sample_triples,
            num_entities=100,
            num_negatives=2,  # 2 negative samples
        )

        item = dataset[0]

        # Should return dict with 'positive' and 'negatives'
        assert isinstance(item, dict)
        assert 'positive' in item
        assert 'negatives' in item

        positive = item['positive']
        negatives = item['negatives']

        assert positive.shape == (3,)  # (h, r, t)
        assert negatives.shape == (2, 3)  # (num_negatives, 3)

    def test_dataset_negative_sampling_valid(self, sample_triples):
        """Test negative samples are valid entity indices."""
        dataset = TransEDataset(
            triples=sample_triples,
            num_entities=100,
            num_negatives=5,
        )

        for i in range(len(dataset)):
            item = dataset[i]
            positive = item['positive']
            negatives = item['negatives']

            # Check all negative samples have valid entity indices
            assert torch.all(negatives[:, 0] >= 0) and torch.all(negatives[:, 0] < 100)  # heads
            assert torch.all(negatives[:, 2] >= 0) and torch.all(negatives[:, 2] < 100)  # tails

    def test_dataset_deterministic_with_seed(self, sample_triples):
        """Test dataset is deterministic with same seed."""
        dataset1 = TransEDataset(sample_triples, num_entities=100, num_negatives=1, seed=42)
        dataset2 = TransEDataset(sample_triples, num_entities=100, num_negatives=1, seed=42)

        item1 = dataset1[0]
        item2 = dataset2[0]

        # Positive triples should be identical
        assert torch.equal(item1['positive'], item2['positive'])

        # Negative triples should be identical (same seed)
        assert torch.equal(item1['negatives'], item2['negatives'])


# ═══════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestTransEIntegration:
    """Test TransE model integration (forward + backward)."""

    def test_model_trainable(self, transe_model, sample_triples):
        """Test model can perform forward and backward pass."""
        # Create simple training batch
        heads = torch.tensor([0, 1, 2])
        relations = torch.tensor([0, 1, 0])
        tails = torch.tensor([1, 2, 3])

        # Forward pass
        pos_scores = transe_model.forward(heads, relations, tails)

        # Create negative samples (corrupt tails)
        neg_tails = torch.tensor([50, 60, 70])
        neg_scores = transe_model.forward(heads, relations, neg_tails)

        # Compute margin ranking loss (simplified)
        loss = torch.mean(torch.clamp(transe_model.margin + neg_scores - pos_scores, min=0))

        # Backward pass should work
        loss.backward()

        # Gradients should exist
        assert transe_model.entity_embeddings.weight.grad is not None
        assert transe_model.relation_embeddings.weight.grad is not None

    def test_model_parameters_update(self, transe_model):
        """Test model parameters update during training."""
        # Get initial parameters
        initial_entity_emb = transe_model.entity_embeddings.weight.data.clone()

        # Simple training step
        optimizer = torch.optim.Adam(transe_model.parameters(), lr=0.01)

        heads = torch.tensor([0])
        relations = torch.tensor([0])
        pos_tails = torch.tensor([1])
        neg_tails = torch.tensor([50])

        pos_scores = transe_model.forward(heads, relations, pos_tails)
        neg_scores = transe_model.forward(heads, relations, neg_tails)

        loss = torch.mean(torch.clamp(transe_model.margin + neg_scores - pos_scores, min=0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Parameters should have changed
        assert not torch.equal(initial_entity_emb, transe_model.entity_embeddings.weight.data)

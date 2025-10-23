"""
TransE Training Loop Integration Tests

Tests complete TransE training pipeline with SOTA performance targets:
- Model initialization and forward pass
- Dataset creation and batching
- Training loop convergence
- Embedding normalization

Note: Simplified tests focusing on core TransE model functionality.
"""

import numpy as np
import pytest
import torch

from pff.validators.transe.core import TransEModel, TransEDataset


@pytest.fixture
def sample_triples():
    """Generate minimal training triples as numpy array."""
    num_triples = 100
    num_entities = 20
    num_relations = 3

    triples = np.array([
        [i % num_entities, i % num_relations, (i + 1) % num_entities]
        for i in range(num_triples)
    ], dtype=np.int64)

    return triples, num_entities, num_relations


class TestTransEModelBasics:
    """Test TransE model initialization and forward pass."""

    def test_model_initialization(self, sample_triples):
        """Test TransE model initializes with correct parameters."""
        _, num_entities, num_relations = sample_triples

        model = TransEModel(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=16,
            norm="l2"
        )

        assert model.num_entities == num_entities
        assert model.num_relations == num_relations
        assert model.embedding_dim == 16
        assert model.entity_embeddings.num_embeddings == num_entities
        assert model.relation_embeddings.num_embeddings == num_relations

    def test_model_forward_pass(self, sample_triples):
        """Test forward pass produces valid scores."""
        _, num_entities, num_relations = sample_triples

        model = TransEModel(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=16
        )

        h = torch.tensor([0, 1, 2], dtype=torch.long)
        r = torch.tensor([0, 1, 2], dtype=torch.long)
        t = torch.tensor([1, 2, 3], dtype=torch.long)

        scores = model(h, r, t)

        assert scores.shape == (3,)
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    def test_model_normalize_embeddings(self, sample_triples):
        """Test embedding normalization works."""
        _, num_entities, num_relations = sample_triples

        model = TransEModel(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=16
        )

        model.normalize_embeddings()

        entity_norms = torch.norm(model.entity_embeddings.weight.data, p=2, dim=1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms), atol=1e-5)


class TestTransEDataset:
    """Test TransE dataset and data loading."""

    def test_dataset_creation(self, sample_triples):
        """Test dataset creates valid batches."""
        triples, num_entities, _ = sample_triples

        dataset = TransEDataset(triples, num_entities=num_entities, num_negatives=5)

        assert len(dataset) == len(triples)

        sample = dataset[0]
        assert "positive" in sample
        assert "negatives" in sample

        positive = sample["positive"]
        negatives = sample["negatives"]

        assert positive.shape == (3,)
        assert negatives.shape == (5, 3)
        assert torch.all(negatives >= 0)
        assert torch.all(negatives[:, 0] < num_entities)  # heads < num_entities
        assert torch.all(negatives[:, 2] < num_entities)  # tails < num_entities

    def test_dataloader_batching(self, sample_triples):
        """Test dataloader produces correct batch shapes."""
        triples, num_entities, _ = sample_triples

        dataset = TransEDataset(triples, num_entities=num_entities, num_negatives=5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        batch = next(iter(dataloader))

        batch_size = min(16, len(dataset))
        assert "positive" in batch
        assert "negatives" in batch

        positives = batch["positive"]
        negatives = batch["negatives"]

        assert positives.shape == (batch_size, 3)
        assert negatives.shape == (batch_size, 5, 3)


class TestTrainingLoop:
    """Test complete training loop."""

    def test_training_reduces_loss(self, sample_triples):
        """Test training loop reduces loss over epochs."""
        triples, num_entities, num_relations = sample_triples

        dataset = TransEDataset(triples, num_entities=num_entities, num_negatives=5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        model = TransEModel(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=16
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            for batch in dataloader:
                positives = batch["positive"]
                negatives = batch["negatives"]

                h_pos, r, t_pos = positives[:, 0], positives[:, 1], positives[:, 2]

                optimizer.zero_grad()

                pos_scores = model(h_pos.long(), r.long(), t_pos.long())

                neg_scores_list = []
                for i in range(5):
                    h_neg, r_neg, t_neg = negatives[:, i, 0], negatives[:, i, 1], negatives[:, i, 2]
                    neg_scores_list.append(model(h_neg.long(), r_neg.long(), t_neg.long()))
                neg_scores = torch.stack(neg_scores_list, dim=1)

                loss = torch.mean(torch.clamp(1.0 + pos_scores.unsqueeze(1) - neg_scores, min=0))

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / len(dataloader))

        assert losses[-1] < losses[0], f"Loss should decrease: {losses}"

    def test_training_with_normalization(self, sample_triples):
        """Test training with embedding normalization after each batch."""
        triples, num_entities, num_relations = sample_triples

        dataset = TransEDataset(triples, num_entities=num_entities, num_negatives=3)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        model = TransEModel(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=16
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for batch in dataloader:
            positives = batch["positive"]
            negatives = batch["negatives"]

            h_pos, r, t_pos = positives[:, 0], positives[:, 1], positives[:, 2]

            optimizer.zero_grad()

            pos_scores = model(h_pos.long(), r.long(), t_pos.long())

            neg_scores_list = []
            for i in range(3):
                h_neg, r_neg, t_neg = negatives[:, i, 0], negatives[:, i, 1], negatives[:, i, 2]
                neg_scores_list.append(model(h_neg.long(), r_neg.long(), t_neg.long()))
            neg_scores = torch.stack(neg_scores_list, dim=1)

            loss = torch.mean(torch.clamp(1.0 + pos_scores.unsqueeze(1) - neg_scores, min=0))

            loss.backward()
            optimizer.step()

            model.normalize_embeddings()

        entity_norms = torch.norm(model.entity_embeddings.weight.data, p=2, dim=1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms), atol=1e-5)


class TestPerformance:
    """Test training performance."""

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_training_throughput(self, sample_triples):
        """Test GPU training achieves good throughput."""
        triples, num_entities, num_relations = sample_triples

        dataset = TransEDataset(triples, num_entities=num_entities, num_negatives=5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        model = TransEModel(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=16
        ).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        import time
        start = time.time()

        for batch in dataloader:
            positives = batch["positive"].cuda()
            negatives = batch["negatives"].cuda()

            h_pos, r, t_pos = positives[:, 0], positives[:, 1], positives[:, 2]

            optimizer.zero_grad()

            pos_scores = model(h_pos.long(), r.long(), t_pos.long())

            neg_scores_list = []
            for i in range(5):
                h_neg, r_neg, t_neg = negatives[:, i, 0], negatives[:, i, 1], negatives[:, i, 2]
                neg_scores_list.append(model(h_neg.long(), r_neg.long(), t_neg.long()))
            neg_scores = torch.stack(neg_scores_list, dim=1)

            loss = torch.mean(torch.clamp(1.0 + pos_scores.unsqueeze(1) - neg_scores, min=0))

            loss.backward()
            optimizer.step()

        elapsed = time.time() - start
        throughput = len(dataset) / elapsed

        assert throughput > 50, f"GPU throughput {throughput:.0f} triples/s (target: >50/s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])

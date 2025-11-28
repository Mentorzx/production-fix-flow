"""Tests for attention mechanism in neural-symbolic ensemble.

Author: PFF Team
Date: 2025-11-25
"""

import numpy as np
import pytest
import torch

from pff.validators.ensembles.attention import (
    AttentionConfig,
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeatureTypeAttention,
    AttentionEnsembleTransformer,
    AttentionFactory,
)


class TestAttentionConfig:
    """Tests for AttentionConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = AttentionConfig()
        
        assert config.hidden_dim == 64
        assert config.num_heads == 4
        assert config.dropout == 0.1
        assert config.temperature == 1.0
        assert config.use_layer_norm is True
        assert config.attention_type == "scaled_dot"
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = AttentionConfig(
            hidden_dim=128,
            num_heads=8,
            dropout=0.2,
            temperature=0.5,
        )
        
        assert config.hidden_dim == 128
        assert config.num_heads == 8
        assert config.dropout == 0.2
        assert config.temperature == 0.5


class TestScaledDotProductAttention:
    """Tests for ScaledDotProductAttention."""
    
    def test_forward_pass(self):
        """Test forward pass shape."""
        config = AttentionConfig(hidden_dim=32)
        attention = ScaledDotProductAttention(config, input_dim=16)
        
        batch_size, seq_len, dim = 4, 8, 16
        x = torch.randn(batch_size, seq_len, dim)
        
        output, weights = attention(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert weights.shape == (batch_size, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        config = AttentionConfig(hidden_dim=32, dropout=0.0)
        attention = ScaledDotProductAttention(config, input_dim=16)
        attention.eval()
        
        x = torch.randn(2, 4, 16)
        
        with torch.no_grad():
            _, weights = attention(x)
        
        sums = weights.sum(dim=-1)
        np.testing.assert_allclose(sums.numpy(), np.ones_like(sums.numpy()), atol=1e-5)
    
    def test_with_mask(self):
        """Test attention with mask."""
        config = AttentionConfig(hidden_dim=32, dropout=0.0)
        attention = ScaledDotProductAttention(config, input_dim=16)
        attention.eval()
        
        x = torch.randn(2, 4, 16)
        mask = torch.ones(2, 4, 4)
        mask[:, :, 2:] = 0
        
        with torch.no_grad():
            _, weights = attention(x, mask=mask)
        
        assert (weights[:, :, 2:] < 1e-6).all()


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""
    
    def test_forward_pass(self):
        """Test forward pass shape."""
        config = AttentionConfig(hidden_dim=32, num_heads=4)
        attention = MultiHeadAttention(config, input_dim=16)
        
        batch_size, seq_len, dim = 4, 8, 16
        x = torch.randn(batch_size, seq_len, dim)
        
        output, weights = attention(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert weights.shape == (batch_size, seq_len, seq_len)
    
    def test_num_heads_constraint(self):
        """Test that hidden_dim must be divisible by num_heads."""
        config = AttentionConfig(hidden_dim=30, num_heads=4)
        
        with pytest.raises(AssertionError):
            MultiHeadAttention(config, input_dim=16)
    
    def test_multi_head_averages_weights(self):
        """Test that output weights are averaged across heads."""
        config = AttentionConfig(hidden_dim=32, num_heads=4, dropout=0.0)
        attention = MultiHeadAttention(config, input_dim=16)
        attention.eval()
        
        x = torch.randn(2, 4, 16)
        
        with torch.no_grad():
            _, weights = attention(x)
        
        sums = weights.sum(dim=-1)
        np.testing.assert_allclose(sums.numpy(), np.ones_like(sums.numpy()), atol=1e-5)


class TestFeatureTypeAttention:
    """Tests for FeatureTypeAttention."""
    
    def test_forward_pass(self):
        """Test forward pass shape."""
        config = AttentionConfig(hidden_dim=32)
        attention = FeatureTypeAttention(
            neural_dim=10,
            symbolic_dim=5,
            hybrid_dim=3,
            config=config,
        )
        
        batch_size = 8
        neural = torch.randn(batch_size, 10)
        symbolic = torch.randn(batch_size, 5)
        hybrid = torch.randn(batch_size, 3)
        
        output, weights = attention(neural, symbolic, hybrid)
        
        assert output.shape == (batch_size, 18)
        assert weights.shape == (batch_size, 3)
    
    def test_attention_weights_sum_to_one(self):
        """Test that type attention weights sum to 1."""
        config = AttentionConfig(hidden_dim=32, dropout=0.0)
        attention = FeatureTypeAttention(
            neural_dim=10,
            symbolic_dim=5,
            hybrid_dim=3,
            config=config,
        )
        attention.eval()
        
        neural = torch.randn(4, 10)
        symbolic = torch.randn(4, 5)
        hybrid = torch.randn(4, 3)
        
        with torch.no_grad():
            _, weights = attention(neural, symbolic, hybrid)
        
        sums = weights.sum(dim=-1)
        np.testing.assert_allclose(sums.numpy(), np.ones(4), atol=1e-5)
    
    def test_get_attention_stats(self):
        """Test attention statistics computation."""
        attention = FeatureTypeAttention(
            neural_dim=10,
            symbolic_dim=5,
            hybrid_dim=3,
        )
        
        neural = torch.randn(4, 10)
        symbolic = torch.randn(4, 5)
        hybrid = torch.randn(4, 3)
        
        _, weights = attention(neural, symbolic, hybrid)
        stats = attention.get_attention_stats(weights)
        
        assert "neural_mean" in stats
        assert "symbolic_mean" in stats
        assert "hybrid_mean" in stats
        assert 0 <= stats["neural_mean"] <= 1
        assert 0 <= stats["symbolic_mean"] <= 1
        assert 0 <= stats["hybrid_mean"] <= 1


class TestAttentionEnsembleTransformer:
    """Tests for AttentionEnsembleTransformer."""
    
    def test_fit_transform(self):
        """Test fit and transform operations."""
        np.random.seed(42)
        
        n_samples = 100
        X = np.random.randn(n_samples, 15)
        y = np.random.randint(0, 2, n_samples)
        
        feature_names = (
            [f"neural_{i}" for i in range(5)] +
            [f"rule_{i}" for i in range(5)] +
            [f"hybrid_{i}" for i in range(5)]
        )
        
        transformer = AttentionEnsembleTransformer(
            epochs=2,
            batch_size=32,
        )
        
        X_transformed = transformer.fit_transform(X, y, feature_names=feature_names)
        
        assert X_transformed.shape[0] == n_samples
        assert X_transformed.shape[1] > X.shape[1]
    
    def test_get_attention_weights(self):
        """Test retrieving attention weights."""
        np.random.seed(42)
        
        X = np.random.randn(50, 12)
        y = np.random.randint(0, 2, 50)
        
        feature_names = (
            [f"neural_{i}" for i in range(4)] +
            [f"rule_{i}" for i in range(4)] +
            [f"hybrid_{i}" for i in range(4)]
        )
        
        transformer = AttentionEnsembleTransformer(epochs=2, batch_size=16)
        transformer.fit(X, y, feature_names=feature_names)
        
        weights = transformer.get_attention_weights(X)
        
        assert weights.shape == (50, 3)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(50), atol=1e-5)
    
    def test_get_feature_names_out(self):
        """Test output feature names."""
        X = np.random.randn(20, 6)
        y = np.random.randint(0, 2, 20)
        
        feature_names = ["neural_1", "neural_2", "rule_1", "rule_2", "hybrid_1", "hybrid_2"]
        
        transformer = AttentionEnsembleTransformer(epochs=1, batch_size=10)
        transformer.fit(X, y, feature_names=feature_names)
        
        out_names = transformer.get_feature_names_out()
        
        assert len(out_names) > 6
        assert all(name in out_names for name in feature_names)
        assert any("attention_weighted" in name for name in out_names)
    
    def test_get_attention_statistics(self):
        """Test attention statistics retrieval."""
        X = np.random.randn(20, 6)
        y = np.random.randint(0, 2, 20)
        
        transformer = AttentionEnsembleTransformer(epochs=3, batch_size=10)
        transformer.fit(X, y)
        
        stats = transformer.get_attention_statistics()
        
        assert "training_history" in stats
        assert "config" in stats
        assert len(stats["training_history"]) == 3


class TestAttentionFactory:
    """Tests for AttentionFactory."""
    
    def test_create_scaled_dot(self):
        """Test creating scaled dot attention."""
        config = AttentionConfig(hidden_dim=32)
        attention = AttentionFactory.create("scaled_dot", config, input_dim=16)
        
        assert isinstance(attention, ScaledDotProductAttention)
    
    def test_create_multi_head(self):
        """Test creating multi-head attention."""
        config = AttentionConfig(hidden_dim=32, num_heads=4)
        attention = AttentionFactory.create("multi_head", config, input_dim=16)
        
        assert isinstance(attention, MultiHeadAttention)
    
    def test_create_unknown_type(self):
        """Test error on unknown attention type."""
        config = AttentionConfig()
        
        with pytest.raises(ValueError, match="Unknown attention type"):
            AttentionFactory.create("unknown", config, input_dim=16)
    
    def test_list_available(self):
        """Test listing available attention types."""
        available = AttentionFactory.list_available()
        
        assert "scaled_dot" in available
        assert "multi_head" in available
    
    def test_register_custom(self):
        """Test registering custom attention type."""
        class CustomAttention(ScaledDotProductAttention):
            pass
        
        AttentionFactory.register("custom", CustomAttention)
        
        assert "custom" in AttentionFactory.list_available()
        
        config = AttentionConfig(hidden_dim=32)
        attention = AttentionFactory.create("custom", config, input_dim=16)
        
        assert isinstance(attention, CustomAttention)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

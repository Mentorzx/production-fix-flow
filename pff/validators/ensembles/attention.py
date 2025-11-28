"""Attention mechanism for neural-symbolic ensemble synergy.

This module provides attention layers that learn to weight the importance
of different feature types (neural, symbolic, hybrid) dynamically based
on input characteristics.

Design Patterns Applied:
    - **Strategy Pattern:** Different attention types (self, cross, multi-head).
    - **Adapter Pattern:** PyTorch attention adapted to sklearn interface.
    - **Observer Pattern:** Training metrics notification.

Reference Papers:
    - Vaswani et al., "Attention Is All You Need" (2017)
    - Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align
      and Translate" (2015)
    - Knowledge Graph + Attention: "KGAT: Knowledge Graph Attention Network" (2019)

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Protocol
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class AttentionConfig:
    """Configuration for attention mechanism.
    
    Attributes:
        hidden_dim: Hidden dimension for projections (default: 64).
        num_heads: Number of attention heads for multi-head attention (default: 4).
        dropout: Dropout probability (default: 0.1).
        temperature: Softmax temperature for attention weights (default: 1.0).
        use_layer_norm: Whether to apply layer normalization (default: True).
        attention_type: Type of attention - 'scaled_dot', 'additive', 'multi_head'.
    """
    hidden_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    temperature: float = 1.0
    use_layer_norm: bool = True
    attention_type: str = "scaled_dot"


class AttentionObserver(Protocol):
    """Protocol for attention training observers."""
    
    def on_attention_computed(
        self, 
        weights: np.ndarray, 
        feature_groups: list[str],
        batch_idx: int,
    ) -> None:
        """Called when attention weights are computed."""
        ...


class BaseAttention(nn.Module, ABC):
    """Abstract base class for attention mechanisms.
    
    Template Method Pattern: Defines the skeleton of attention computation.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self._observers: list[AttentionObserver] = []
    
    def add_observer(self, observer: AttentionObserver) -> None:
        """Add an observer for attention events."""
        self._observers.append(observer)
    
    def remove_observer(self, observer: AttentionObserver) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(
        self, 
        weights: np.ndarray, 
        feature_groups: list[str],
        batch_idx: int,
    ) -> None:
        """Notify observers of attention computation."""
        for observer in self._observers:
            observer.on_attention_computed(weights, feature_groups, batch_idx)
    
    @abstractmethod
    def compute_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and output.
        
        Args:
            query: Query tensor [batch, seq_len, dim].
            key: Key tensor [batch, seq_len, dim].
            value: Value tensor [batch, seq_len, dim].
            mask: Optional attention mask.
            
        Returns:
            Tuple of (attended_output, attention_weights).
        """
        pass


class ScaledDotProductAttention(BaseAttention):
    """Scaled dot-product attention mechanism.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    
    def __init__(self, config: AttentionConfig, input_dim: int):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.scale = math.sqrt(config.hidden_dim)
        
        self.q_proj = nn.Linear(input_dim, config.hidden_dim)
        self.k_proj = nn.Linear(input_dim, config.hidden_dim)
        self.v_proj = nn.Linear(input_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, input_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)
        else:
            self.layer_norm = None
    
    def compute_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention."""
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores / self.config.temperature
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        output = self.out_proj(context)
        
        if self.layer_norm is not None:
            output = self.layer_norm(output + query)
        
        return output, attention_weights
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with self-attention."""
        return self.compute_attention(x, x, x, mask)


class MultiHeadAttention(BaseAttention):
    """Multi-head attention mechanism.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
    where head_i = Attention(Q * W^Q_i, K * W^K_i, V * W^V_i)
    """
    
    def __init__(self, config: AttentionConfig, input_dim: int):
        super().__init__(config)
        
        assert config.hidden_dim % config.num_heads == 0, \
            f"hidden_dim ({config.hidden_dim}) must be divisible by num_heads ({config.num_heads})"
        
        self.input_dim = input_dim
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(input_dim, config.hidden_dim)
        self.k_proj = nn.Linear(input_dim, config.hidden_dim)
        self.v_proj = nn.Linear(input_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, input_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)
        else:
            self.layer_norm = None
    
    def compute_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention."""
        batch_size = query.size(0)
        
        Q = self.q_proj(query).view(batch_size, -1, self.config.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, -1, self.config.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, -1, self.config.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores / self.config.temperature
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.hidden_dim)
        output = self.out_proj(context)
        
        if self.layer_norm is not None:
            output = self.layer_norm(output + query)
        
        avg_weights = attention_weights.mean(dim=1)
        
        return output, avg_weights
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with self-attention."""
        return self.compute_attention(x, x, x, mask)


class FeatureTypeAttention(nn.Module):
    """Attention mechanism that learns to weight feature types.
    
    Groups features into neural, symbolic, and hybrid categories
    and learns attention weights for each group dynamically.
    """
    
    def __init__(
        self, 
        neural_dim: int,
        symbolic_dim: int,
        hybrid_dim: int,
        config: Optional[AttentionConfig] = None,
    ):
        super().__init__()
        
        self.config = config or AttentionConfig()
        self.neural_dim = neural_dim
        self.symbolic_dim = symbolic_dim
        self.hybrid_dim = hybrid_dim
        self.total_dim = neural_dim + symbolic_dim + hybrid_dim
        
        self.neural_proj = nn.Linear(neural_dim, self.config.hidden_dim)
        self.symbolic_proj = nn.Linear(symbolic_dim, self.config.hidden_dim)
        self.hybrid_proj = nn.Linear(hybrid_dim, self.config.hidden_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 3),
        )
        
        self.output_proj = nn.Linear(self.config.hidden_dim, self.total_dim)
        
        if self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.config.hidden_dim)
    
    def forward(
        self, 
        neural_feats: torch.Tensor,
        symbolic_feats: torch.Tensor,
        hybrid_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attended features with type-specific attention.
        
        Args:
            neural_feats: Neural features [batch, neural_dim].
            symbolic_feats: Symbolic features [batch, symbolic_dim].
            hybrid_feats: Hybrid features [batch, hybrid_dim].
            
        Returns:
            Tuple of (attended_features, attention_weights).
        """
        h_neural = self.neural_proj(neural_feats)
        h_symbolic = self.symbolic_proj(symbolic_feats)
        h_hybrid = self.hybrid_proj(hybrid_feats)
        
        concat = torch.cat([h_neural, h_symbolic, h_hybrid], dim=-1)
        
        gate_input = concat
        attention_logits = self.gate(gate_input)
        attention_weights = F.softmax(
            attention_logits / self.config.temperature, dim=-1
        )
        
        weighted = (
            attention_weights[:, 0:1] * h_neural +
            attention_weights[:, 1:2] * h_symbolic +
            attention_weights[:, 2:3] * h_hybrid
        )
        
        if self.config.use_layer_norm:
            weighted = self.layer_norm(weighted)
        
        output = self.output_proj(weighted)
        
        return output, attention_weights
    
    def get_attention_stats(self, attention_weights: torch.Tensor) -> dict[str, float]:
        """Get statistics about attention distribution."""
        weights_np = attention_weights.detach().cpu().numpy()
        
        return {
            "neural_mean": float(weights_np[:, 0].mean()),
            "neural_std": float(weights_np[:, 0].std()),
            "symbolic_mean": float(weights_np[:, 1].mean()),
            "symbolic_std": float(weights_np[:, 1].std()),
            "hybrid_mean": float(weights_np[:, 2].mean()),
            "hybrid_std": float(weights_np[:, 2].std()),
        }


class AttentionEnsembleTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer that applies attention to ensemble features.
    
    This transformer learns to weight different feature types (neural, symbolic,
    hybrid) using an attention mechanism, enabling the model to focus on the
    most relevant features for each sample.
    
    Attributes:
        neural_feature_prefix: Prefix for neural feature names.
        symbolic_feature_prefix: Prefix for symbolic feature names.
        config: Attention configuration.
        
    Example:
        >>> transformer = AttentionEnsembleTransformer()
        >>> X_transformed = transformer.fit_transform(X, y)
    """
    
    def __init__(
        self,
        neural_feature_prefix: str = "neural_",
        symbolic_feature_prefix: str = "rule_",
        config: Optional[AttentionConfig] = None,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        device: Optional[str] = None,
    ):
        self.neural_feature_prefix = neural_feature_prefix
        self.symbolic_feature_prefix = symbolic_feature_prefix
        self.config = config or AttentionConfig()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.attention_module_: Optional[FeatureTypeAttention] = None
        self.feature_names_in_: Optional[list[str]] = None
        self._neural_mask: Optional[np.ndarray] = None
        self._symbolic_mask: Optional[np.ndarray] = None
        self._hybrid_mask: Optional[np.ndarray] = None
        self._attention_history: list[dict[str, float]] = []
    
    def _parse_feature_groups(self, feature_names: list[str]) -> None:
        """Parse feature names into neural, symbolic, hybrid groups."""
        n_features = len(feature_names)
        
        self._neural_mask = np.array([
            name.startswith(self.neural_feature_prefix)
            for name in feature_names
        ])
        self._symbolic_mask = np.array([
            name.startswith(self.symbolic_feature_prefix)
            for name in feature_names
        ])
        self._hybrid_mask = ~(self._neural_mask | self._symbolic_mask)
        
        self._neural_indices = np.where(self._neural_mask)[0]
        self._symbolic_indices = np.where(self._symbolic_mask)[0]
        self._hybrid_indices = np.where(self._hybrid_mask)[0]
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "AttentionEnsembleTransformer":
        """Fit the attention mechanism on training data.
        
        Args:
            X: Feature matrix [n_samples, n_features].
            y: Target labels.
            feature_names: Optional list of feature names.
            
        Returns:
            Self.
        """
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        self.feature_names_in_ = list(feature_names)
        self._parse_feature_groups(self.feature_names_in_)
        
        neural_dim = max(1, len(self._neural_indices))
        symbolic_dim = max(1, len(self._symbolic_indices))
        hybrid_dim = max(1, len(self._hybrid_indices))
        
        self.attention_module_ = FeatureTypeAttention(
            neural_dim=neural_dim,
            symbolic_dim=symbolic_dim,
            hybrid_dim=hybrid_dim,
            config=self.config,
        ).to(self.device)
        
        self._train_attention(X, y)
        
        return self
    
    def _train_attention(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the attention module."""
        self.attention_module_.train()
        optimizer = torch.optim.Adam(
            self.attention_module_.parameters(), 
            lr=self.learning_rate,
        )
        
        n_samples = X.shape[0]
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                neural_feats = self._extract_group(X_batch, self._neural_indices)
                symbolic_feats = self._extract_group(X_batch, self._symbolic_indices)
                hybrid_feats = self._extract_group(X_batch, self._hybrid_indices)
                
                neural_t = torch.tensor(neural_feats, dtype=torch.float32, device=self.device)
                symbolic_t = torch.tensor(symbolic_feats, dtype=torch.float32, device=self.device)
                hybrid_t = torch.tensor(hybrid_feats, dtype=torch.float32, device=self.device)
                y_t = torch.tensor(y_batch, dtype=torch.float32, device=self.device)
                
                optimizer.zero_grad()
                output, attention_weights = self.attention_module_(
                    neural_t, symbolic_t, hybrid_t
                )
                
                logits = output.mean(dim=-1)
                loss = F.binary_cross_entropy_with_logits(logits, y_t)
                
                entropy = -(attention_weights * (attention_weights + 1e-8).log()).sum(dim=-1).mean()
                loss = loss - 0.01 * entropy
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            self._attention_history.append({
                "epoch": epoch,
                "loss": epoch_loss,
            })
        
        self.attention_module_.eval()
    
    def _extract_group(
        self, 
        X: np.ndarray, 
        indices: np.ndarray,
    ) -> np.ndarray:
        """Extract feature group from X."""
        if len(indices) == 0:
            return np.zeros((X.shape[0], 1))
        return X[:, indices]
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using learned attention.
        
        Args:
            X: Feature matrix [n_samples, n_features].
            
        Returns:
            Transformed features with attention-weighted combinations.
        """
        if self.attention_module_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        self.attention_module_.eval()
        
        neural_feats = self._extract_group(X, self._neural_indices)
        symbolic_feats = self._extract_group(X, self._symbolic_indices)
        hybrid_feats = self._extract_group(X, self._hybrid_indices)
        
        with torch.no_grad():
            neural_t = torch.tensor(neural_feats, dtype=torch.float32, device=self.device)
            symbolic_t = torch.tensor(symbolic_feats, dtype=torch.float32, device=self.device)
            hybrid_t = torch.tensor(hybrid_feats, dtype=torch.float32, device=self.device)
            
            output, attention_weights = self.attention_module_(
                neural_t, symbolic_t, hybrid_t
            )
            
            self._last_attention_weights = attention_weights.cpu().numpy()
        
        X_out = np.concatenate([X, output.cpu().numpy()], axis=1)
        
        return X_out
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names."""
        if input_features is None:
            input_features = self.feature_names_in_ or []
        
        attention_features = [f"attention_weighted_{i}" for i in range(self.attention_module_.total_dim)]
        
        return list(input_features) + attention_features
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Get attention weights for input samples.
        
        Returns:
            Array of shape [n_samples, 3] with weights for
            (neural, symbolic, hybrid) feature groups.
        """
        self.transform(X)
        return self._last_attention_weights
    
    def get_attention_statistics(self) -> dict[str, Any]:
        """Get statistics about learned attention."""
        if self.attention_module_ is None:
            return {}
        
        return {
            "training_history": self._attention_history,
            "config": {
                "hidden_dim": self.config.hidden_dim,
                "num_heads": self.config.num_heads,
                "dropout": self.config.dropout,
                "temperature": self.config.temperature,
            },
        }


class AttentionFactory:
    """Factory for creating attention mechanisms.
    
    Factory Pattern: Centralized creation of attention instances.
    """
    
    _registry: dict[str, type[BaseAttention]] = {
        "scaled_dot": ScaledDotProductAttention,
        "multi_head": MultiHeadAttention,
    }
    
    @classmethod
    def register(cls, name: str, attention_class: type[BaseAttention]) -> None:
        """Register a new attention type."""
        cls._registry[name] = attention_class
    
    @classmethod
    def create(
        cls, 
        attention_type: str, 
        config: AttentionConfig,
        input_dim: int,
    ) -> BaseAttention:
        """Create an attention mechanism.
        
        Args:
            attention_type: Type of attention ('scaled_dot', 'multi_head').
            config: Attention configuration.
            input_dim: Input dimension.
            
        Returns:
            Configured attention module.
            
        Raises:
            ValueError: If attention_type is not registered.
        """
        if attention_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown attention type: {attention_type}. "
                f"Available: {available}"
            )
        
        attention_class = cls._registry[attention_type]
        return attention_class(config, input_dim)
    
    @classmethod
    def list_available(cls) -> list[str]:
        """List available attention types."""
        return list(cls._registry.keys())


__all__ = [
    "AttentionConfig",
    "AttentionObserver",
    "BaseAttention",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "FeatureTypeAttention",
    "AttentionEnsembleTransformer",
    "AttentionFactory",
]

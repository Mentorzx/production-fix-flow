"""RotatE Configuration Module.

Provides configuration dataclass and builder for RotatE model parameters.
Configuration is loaded from config/models/rotate.yaml via FileManager.

Design Patterns Applied:
    - **Builder Pattern:** RotatEConfigBuilder allows step-by-step configuration.
    - **Factory Method:** from_yaml() creates config from file.

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pff.utils import FileManager, logger


@dataclass
class RotatEConfig:
    """Configuration for RotatE model.

    Attributes:
        embedding_dim: Dimension of entity embeddings (must be even for complex repr).
        gamma: Fixed margin for scoring function (typically 9-24 for RotatE).
        epsilon: Modulus regularization parameter.
        learning_rate: Initial learning rate.
        batch_size: Training batch size.
        num_negatives: Negative samples per positive triple.
        use_self_adversarial: Enable self-adversarial negative sampling.
        adversarial_temperature: Temperature for self-adversarial weighting.
        entity_regularizer_weight: L2 regularization for entity embeddings.
        relation_regularizer_weight: L2 regularization for relation phases.
        double_entity_embedding: Use complex (double) entity embeddings.
        epochs: Number of training epochs.
        early_stopping_patience: Patience for early stopping.
        seed: Random seed for reproducibility.
        extra: Additional model-specific parameters.

    Example:
        >>> config = RotatEConfig(embedding_dim=256, gamma=12.0)
        >>> config.complex_dim  # Returns 128 (embedding_dim // 2)
        128
    """

    embedding_dim: int = 256
    gamma: float = 12.0
    epsilon: float = 2.0
    learning_rate: float = 0.00005
    batch_size: int = 1024
    num_negatives: int = 256
    use_self_adversarial: bool = True
    adversarial_temperature: float = 1.0
    entity_regularizer_weight: float = 0.0
    relation_regularizer_weight: float = 0.0
    double_entity_embedding: bool = True
    epochs: int = 200
    early_stopping_patience: int = 30
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.embedding_dim % 2 != 0:
            raise ValueError(
                f"embedding_dim must be even for complex representation, "
                f"got {self.embedding_dim}"
            )
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

    @property
    def complex_dim(self) -> int:
        """Return the complex dimension (half of embedding_dim).

        RotatE stores entities as complex vectors where:
        - First half: real part
        - Second half: imaginary part

        Returns:
            Complex dimension (embedding_dim // 2).
        """
        return self.embedding_dim // 2

    @classmethod
    def from_yaml(cls, path: Path | str) -> RotatEConfig:
        """Create configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            RotatEConfig instance.

        Raises:
            FileNotFoundError: If config file does not exist.
            ValueError: If config contains invalid values.
        """
        file_manager = FileManager()
        config_dict = file_manager.read(Path(path))

        model_cfg = config_dict.get("model", {})
        training_cfg = config_dict.get("training", {})
        reg_cfg = config_dict.get("regularization", {})

        return cls(
            embedding_dim=model_cfg.get("embedding_dim", 256),
            gamma=model_cfg.get("gamma", 12.0),
            epsilon=model_cfg.get("epsilon", 2.0),
            learning_rate=training_cfg.get("learning_rate", 0.00005),
            batch_size=training_cfg.get("batch_size", 1024),
            num_negatives=training_cfg.get("negative_samples", 256),
            use_self_adversarial=training_cfg.get(
                "self_adversarial_negative_sampling", True
            ),
            adversarial_temperature=training_cfg.get("adversarial_temperature", 1.0),
            entity_regularizer_weight=reg_cfg.get("entity_regularizer_weight", 0.0),
            relation_regularizer_weight=reg_cfg.get("relation_regularizer_weight", 0.0),
            double_entity_embedding=model_cfg.get("double_entity_embedding", True),
            epochs=training_cfg.get("epochs", 200),
            early_stopping_patience=training_cfg.get("early_stopping_patience", 30),
            seed=training_cfg.get("seed", 42),
            extra=config_dict,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary with all configuration parameters.
        """
        return {
            "embedding_dim": self.embedding_dim,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_negatives": self.num_negatives,
            "use_self_adversarial": self.use_self_adversarial,
            "adversarial_temperature": self.adversarial_temperature,
            "entity_regularizer_weight": self.entity_regularizer_weight,
            "relation_regularizer_weight": self.relation_regularizer_weight,
            "double_entity_embedding": self.double_entity_embedding,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "seed": self.seed,
        }


class RotatEConfigBuilder:
    """Builder for RotatEConfig with fluent interface.

    Design Pattern: Builder
        - Separates construction from representation.
        - Provides fluent API for step-by-step configuration.

    Example:
        >>> config = (
        ...     RotatEConfigBuilder()
        ...     .with_embedding_dim(512)
        ...     .with_gamma(18.0)
        ...     .with_self_adversarial(temperature=1.5)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._config_dict: dict[str, Any] = {}

    def with_embedding_dim(self, dim: int) -> RotatEConfigBuilder:
        """Set embedding dimension.

        Args:
            dim: Embedding dimension (must be even).

        Returns:
            Self for chaining.
        """
        self._config_dict["embedding_dim"] = dim
        return self

    def with_gamma(self, gamma: float) -> RotatEConfigBuilder:
        """Set margin gamma.

        Args:
            gamma: Fixed margin for scoring (typically 9-24).

        Returns:
            Self for chaining.
        """
        self._config_dict["gamma"] = gamma
        return self

    def with_epsilon(self, epsilon: float) -> RotatEConfigBuilder:
        """Set modulus regularization epsilon.

        Args:
            epsilon: Regularization parameter (typically 2.0).

        Returns:
            Self for chaining.
        """
        self._config_dict["epsilon"] = epsilon
        return self

    def with_learning_rate(self, lr: float) -> RotatEConfigBuilder:
        """Set learning rate.

        Args:
            lr: Initial learning rate.

        Returns:
            Self for chaining.
        """
        self._config_dict["learning_rate"] = lr
        return self

    def with_batch_size(self, batch_size: int) -> RotatEConfigBuilder:
        """Set batch size.

        Args:
            batch_size: Training batch size.

        Returns:
            Self for chaining.
        """
        self._config_dict["batch_size"] = batch_size
        return self

    def with_negative_samples(self, num_negatives: int) -> RotatEConfigBuilder:
        """Set number of negative samples.

        Args:
            num_negatives: Negative samples per positive triple.

        Returns:
            Self for chaining.
        """
        self._config_dict["num_negatives"] = num_negatives
        return self

    def with_self_adversarial(
        self, enabled: bool = True, temperature: float = 1.0
    ) -> RotatEConfigBuilder:
        """Configure self-adversarial negative sampling.

        Args:
            enabled: Enable self-adversarial sampling.
            temperature: Temperature for weighting.

        Returns:
            Self for chaining.
        """
        self._config_dict["use_self_adversarial"] = enabled
        self._config_dict["adversarial_temperature"] = temperature
        return self

    def with_regularization(
        self, entity_weight: float = 0.0, relation_weight: float = 0.0
    ) -> RotatEConfigBuilder:
        """Configure L2 regularization.

        Args:
            entity_weight: Weight for entity embedding regularization.
            relation_weight: Weight for relation phase regularization.

        Returns:
            Self for chaining.
        """
        self._config_dict["entity_regularizer_weight"] = entity_weight
        self._config_dict["relation_regularizer_weight"] = relation_weight
        return self

    def with_training_params(
        self, epochs: int = 200, patience: int = 30, seed: int = 42
    ) -> RotatEConfigBuilder:
        """Configure training parameters.

        Args:
            epochs: Number of training epochs.
            patience: Early stopping patience.
            seed: Random seed.

        Returns:
            Self for chaining.
        """
        self._config_dict["epochs"] = epochs
        self._config_dict["early_stopping_patience"] = patience
        self._config_dict["seed"] = seed
        return self

    def build(self) -> RotatEConfig:
        """Build the RotatEConfig instance.

        Returns:
            Configured RotatEConfig.

        Raises:
            ValueError: If configuration is invalid.
        """
        return RotatEConfig(**self._config_dict)

    def from_yaml(self, path: Path | str) -> RotatEConfigBuilder:
        """Load base configuration from YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Self for chaining with overrides.
        """
        base_config = RotatEConfig.from_yaml(path)
        self._config_dict = base_config.to_dict()
        return self

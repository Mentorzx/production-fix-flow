"""
DSLFM Search Space Module.

Provides a minimal search space factory for DSLFM/PC-only optimization.
Includes adaptive training configuration based on dataset statistics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pff.domain.learning.ml.adaptive_training import (
    compute_adaptive_config,
)


@dataclass
class TuningConfig:
    """Configuration for DSLFM hyperparameter tuning.

    All values are loaded from config/hpo/optimization.yaml via TuningConfigBuilder.
    Do not instantiate directly - use TuningConfigBuilder().build() instead.
    """

    embedding_dim_choices: Sequence[int]
    max_communities_choices: Sequence[int]
    ibp_alpha_low: float
    ibp_alpha_high: float
    batch_size_low: int
    batch_size_high: int
    negative_ratio_low: float
    negative_ratio_high: float
    negative_sample_size_low: int
    negative_sample_size_high: int
    num_global_negatives_low: int
    num_global_negatives_high: int
    adversarial_temperature_low: float
    adversarial_temperature_high: float
    contrastive_temperature_low: float
    contrastive_temperature_high: float
    learning_rate_low: float
    learning_rate_high: float
    lambda_logic_low: float
    lambda_logic_high: float
    kl_weight_low: float
    kl_weight_high: float
    t_norm_choices: Sequence[str]
    attr_hidden_dim_choices: Sequence[int]
    lambda_pc_low: float
    lambda_pc_high: float
    pruning_threshold_low: float
    pruning_threshold_high: float
    rebuild_every_low: int
    rebuild_every_high: int
    max_circuit_depth_choices: Sequence[int]
    lambda_sum_cap: float
    n_trials: int
    timeout_seconds: int
    self_adversarial_choices: Sequence[bool]
    use_bert_default: bool


class TuningConfigBuilder:
    """Builder for DSLFM TuningConfig."""

    def __init__(self, defaults: dict[str, Any] | None = None) -> None:
        """Execute init.



        Args:

            defaults: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        defaults = defaults or {}
        pruning_threshold_low = defaults.get("pruning_threshold_low")
        pruning_threshold_high = defaults.get("pruning_threshold_high")
        rebuild_every_low = defaults.get("rebuild_every_low")
        rebuild_every_high = defaults.get("rebuild_every_high")
        if (
            pruning_threshold_low is None
            or pruning_threshold_high is None
            or rebuild_every_low is None
            or rebuild_every_high is None
        ):
            raise ValueError(
                "Missing PC bounds in optimization config: pruning_threshold_* or rebuild_every_*"
            )

        self._config: dict[str, Any] = {
            "embedding_dim_choices": defaults.get("embedding_dim_choices", (128, 256)),
            "max_communities_choices": defaults.get("max_communities_choices", (64, 128)),
            "ibp_alpha_low": defaults.get("ibp_alpha_low", 1.0),
            "ibp_alpha_high": defaults.get("ibp_alpha_high", 10.0),
            "batch_size_low": defaults.get("batch_size_low", 192),
            "batch_size_high": defaults.get("batch_size_high", 512),
            "negative_ratio_low": defaults.get("negative_ratio_low", 0.4),
            "negative_ratio_high": defaults.get("negative_ratio_high", 0.8),
            "negative_sample_size_low": defaults.get("negative_sample_size_low", 64),
            "negative_sample_size_high": defaults.get("negative_sample_size_high", 512),
            "num_global_negatives_low": defaults.get("num_global_negatives_low", 64),
            "num_global_negatives_high": defaults.get("num_global_negatives_high", 256),
            "adversarial_temperature_low": defaults.get("adversarial_temperature_low", 0.5),
            "adversarial_temperature_high": defaults.get("adversarial_temperature_high", 2.0),
            "contrastive_temperature_low": defaults.get("contrastive_temperature_low", 0.1),
            "contrastive_temperature_high": defaults.get("contrastive_temperature_high", 1.0),
            "learning_rate_low": defaults.get("learning_rate_low", 5e-5),
            "learning_rate_high": defaults.get("learning_rate_high", 3e-4),
            "lambda_logic_low": defaults.get("lambda_logic_low", 0.0),
            "lambda_logic_high": defaults.get("lambda_logic_high", 0.6),
            "kl_weight_low": defaults.get("kl_weight_low", 1e-4),
            "kl_weight_high": defaults.get("kl_weight_high", 5e-2),
            "t_norm_choices": tuple(defaults.get("t_norm_choices", ("product", "lukasiewicz"))),
            "attr_hidden_dim_choices": tuple(
                defaults.get("attr_hidden_dim_choices", (64, 128, 256))
            ),
            "lambda_pc_low": defaults.get("lambda_pc_low", 0.0),
            "lambda_pc_high": defaults.get("lambda_pc_high", 0.6),
            "pruning_threshold_low": float(pruning_threshold_low),
            "pruning_threshold_high": float(pruning_threshold_high),
            "rebuild_every_low": int(rebuild_every_low),
            "rebuild_every_high": int(rebuild_every_high),
            "max_circuit_depth_choices": tuple(
                defaults.get("max_circuit_depth_choices", (2, 3, 4, 5, 6, 7, 8))
            ),
            "lambda_sum_cap": defaults.get("lambda_sum_cap", 0.7),
            "n_trials": defaults.get("n_trials", 50),
            "timeout_seconds": defaults.get("timeout_seconds", 1800),
            "self_adversarial_choices": tuple(defaults.get("self_adversarial_choices", (False,))),
            "use_bert_default": bool(defaults.get("use_bert", True)),
        }

    def with_embedding_dim_choices(self, choices: Sequence[int]) -> TuningConfigBuilder:
        """Set embedding dimension choices."""
        self._config["embedding_dim_choices"] = tuple(int(c) for c in choices)
        return self

    def with_max_communities_choices(self, choices: Sequence[int]) -> TuningConfigBuilder:
        """Set max communities choices."""
        self._config["max_communities_choices"] = tuple(int(c) for c in choices)
        return self

    def with_ibp_alpha(self, low: float, high: float) -> TuningConfigBuilder:
        """Set IBP alpha bounds."""
        self._config["ibp_alpha_low"] = float(low)
        self._config["ibp_alpha_high"] = float(high)
        return self

    def with_batch_size(self, low: int, high: int) -> TuningConfigBuilder:
        """Set batch size bounds."""
        self._config["batch_size_low"] = low
        self._config["batch_size_high"] = high
        return self

    def with_negative_ratio(self, low: float, high: float) -> TuningConfigBuilder:
        """Set negative sampling ratio bounds."""
        self._config["negative_ratio_low"] = low
        self._config["negative_ratio_high"] = high
        return self

    def build(self) -> TuningConfig:
        """Build the tuning configuration."""
        return TuningConfig(**self._config)


class SearchSpaceFactory:
    """Factory for DSLFM search spaces."""

    @staticmethod
    def create_dslfm_space(config: TuningConfig) -> dict[str, Any]:
        """Create search space dictionary for DSLFM/PC model."""
        return {
            "embedding_dim": list(config.embedding_dim_choices),
            "max_communities": list(config.max_communities_choices),
            "ibp_alpha": (float(config.ibp_alpha_low), float(config.ibp_alpha_high)),
            "batch_size": (int(config.batch_size_low), int(config.batch_size_high)),
            "negative_sample_size": (
                int(config.negative_sample_size_low),
                int(config.negative_sample_size_high),
            ),
            "num_global_negatives": (
                int(config.num_global_negatives_low),
                int(config.num_global_negatives_high),
            ),
            "adversarial_temperature": (
                float(config.adversarial_temperature_low),
                float(config.adversarial_temperature_high),
            ),
            "contrastive_temperature": (
                float(config.contrastive_temperature_low),
                float(config.contrastive_temperature_high),
            ),
            "learning_rate": (
                float(config.learning_rate_low),
                float(config.learning_rate_high),
            ),
            "lambda_logic": (
                float(config.lambda_logic_low),
                float(config.lambda_logic_high),
            ),
            "kl_weight": (float(config.kl_weight_low), float(config.kl_weight_high)),
            "t_norm": list(config.t_norm_choices),
            "attr_hidden_dim": list(config.attr_hidden_dim_choices),
            "lambda_pc": (float(config.lambda_pc_low), float(config.lambda_pc_high)),
            "pruning_threshold": (
                float(config.pruning_threshold_low),
                float(config.pruning_threshold_high),
            ),
            "rebuild_every": (
                int(config.rebuild_every_low),
                int(config.rebuild_every_high),
            ),
            "max_circuit_depth": list(config.max_circuit_depth_choices),
            "lambda_sum_cap": float(config.lambda_sum_cap),
        }

    @staticmethod
    def create_adaptive_training_space(
        num_train_triples: int,
        num_valid_triples: int,
        num_entities: int = 0,
        num_relations: int = 0,
        range_factors: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create adaptive training bounds based on dataset statistics.

        This method computes optimal epoch and early stopping ranges for HPO
        based on dataset characteristics. The bounds are centered around the
        adaptive recommendation.

        Formulas:
            - epochs = base_epochs × entity_factor × relation_factor
              × model_factor × coverage_factor
            - early_stopping_patience = base_patience × stability_factor
            - min_delta = 0.001 × sqrt(10000 / num_valid_triples)

        Args:
            num_train_triples: Number of training triples.
            num_valid_triples: Number of validation triples.
            num_entities: Number of unique entities.
            num_relations: Number of unique relations.
            range_factors: Optional override factors for adaptive bounds.

        Returns:
            Dictionary with HPO bounds for training hyperparameters:
            - epochs: (low, high) tuple
            - early_stopping_patience: (low, high) tuple
            - validate_every: (low, high) tuple
            - min_delta: (low, high) tuple

        Example:
            >>> bounds = SearchSpaceFactory.create_adaptive_training_space(
            ...     num_train_triples=4_898_391,
            ...     num_valid_triples=612_298,
            ...     num_entities=269_889,
            ...     num_relations=44,
            ... )
            >>> bounds["epochs"]
        """
        adaptive = compute_adaptive_config(
            num_train_triples=num_train_triples,
            num_valid_triples=num_valid_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            is_dslfm=True,
        )

        base_epochs = max(40, adaptive.epochs)
        entity_factor = min(
            2.0, max(0.5, (num_entities / 50_000) ** 0.25 if num_entities > 0 else 1.0)
        )
        relation_factor = min(
            2.0, max(0.5, (num_relations / 100) ** 0.25 if num_relations > 0 else 1.0)
        )
        model_factor = 1.0
        coverage_factor = min(
            2.0,
            max(
                0.5,
                (
                    (num_train_triples / max(1, num_entities * max(1, num_relations) * 0.01))
                    if num_entities > 0 and num_relations > 0
                    else 1.0
                ),
            ),
        )
        epochs_adaptive = int(
            base_epochs * entity_factor * relation_factor * model_factor * coverage_factor
        )

        base_patience = max(3, adaptive.early_stopping_patience)
        stability_factor = min(
            1.5,
            max(
                0.5,
                (num_valid_triples / 10_000) ** 0.5 if num_valid_triples > 0 else 1.0,
            ),
        )
        patience_adaptive = int(round(base_patience * stability_factor))

        min_delta_adaptive = 0.001 * ((10_000 / max(1, num_valid_triples)) ** 0.5)

        def _clamp_range(
            low_raw: float, high_raw: float, cap_min: float, cap_max: float
        ) -> tuple[float, float]:
            """Execute clamp range.



            Args:

                low_raw: Input value used by this callable.

                high_raw: Input value used by this callable.

                cap_min: Input value used by this callable.

                cap_max: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

            low = max(cap_min, min(low_raw, cap_max))
            high_candidate = max(low_raw, high_raw)
            high = max(low, min(high_candidate, cap_max))
            if low >= high:
                low = max(cap_min, int(0.5 * cap_max))
                high = cap_max
            return low, high

        def _clamp_float_range(
            low_raw: float, high_raw: float, cap_min: float, cap_max: float
        ) -> tuple[float, float]:
            low = max(cap_min, min(float(low_raw), cap_max))
            high_candidate = max(float(low_raw), float(high_raw))
            high = max(low, min(high_candidate, cap_max))
            if low >= high:
                low = max(cap_min, 0.5 * cap_max)
                high = cap_max
            return float(low), float(high)

        epochs_low, epochs_high = _clamp_range(epochs_adaptive * 0.8, epochs_adaptive * 1.2, 8, 300)

        patience_low, patience_high = _clamp_range(
            patience_adaptive * 0.8, patience_adaptive * 1.2, 5, 25
        )

        validate_low, validate_high = _clamp_range(
            adaptive.validate_every, adaptive.validate_every + 2, 3, 12
        )

        min_delta_low = max(1e-5, min_delta_adaptive * 0.25)
        min_delta_high = max(min_delta_low, min(0.002, min_delta_adaptive))

        range_factors = range_factors or {}
        batch_divisor = int(range_factors.get("batch_size_min_divisor", 4))
        batch_floor = int(range_factors.get("batch_size_min_floor", 64))
        neg_divisor = int(range_factors.get("num_neg_min_divisor", 2))
        neg_floor = int(range_factors.get("num_neg_min_floor", 32))

        patience_low_override = range_factors.get("early_stopping_patience_low")
        patience_high_override = range_factors.get("early_stopping_patience_high")
        if patience_low_override is not None or patience_high_override is not None:
            patience_low, patience_high = _clamp_range(
                float(patience_low_override or patience_low),
                float(patience_high_override or patience_high),
                5,
                25,
            )

        validate_low_override = range_factors.get("validate_every_low")
        validate_high_override = range_factors.get("validate_every_high")
        if validate_low_override is not None or validate_high_override is not None:
            validate_low, validate_high = _clamp_range(
                float(validate_low_override or validate_low),
                float(validate_high_override or validate_high),
                3,
                12,
            )

        min_delta_low_override = range_factors.get("min_delta_low")
        min_delta_high_override = range_factors.get("min_delta_high")
        if min_delta_low_override is not None or min_delta_high_override is not None:
            min_delta_low, min_delta_high = _clamp_float_range(
                float(min_delta_low_override or min_delta_low),
                float(min_delta_high_override or min_delta_high),
                1e-5,
                0.002,
            )

        return {
            "epochs": (epochs_low, epochs_high),
            "epochs_default": epochs_adaptive,
            "early_stopping_patience": (patience_low, patience_high),
            "early_stopping_patience_default": patience_adaptive,
            "validate_every": (validate_low, validate_high),
            "validate_every_default": adaptive.validate_every,
            "min_delta": (min_delta_low, min_delta_high),
            "min_delta_default": min_delta_adaptive,
            "batch_size": (
                max(batch_floor, adaptive.batch_size // batch_divisor),
                adaptive.batch_size,
            ),
            "batch_size_default": adaptive.batch_size,
            "num_neg": (
                max(neg_floor, adaptive.num_neg // neg_divisor),
                adaptive.num_neg,
            ),
            "num_neg_default": adaptive.num_neg,
            "learning_rate_default": adaptive.learning_rate,
            "dataset_scale": adaptive.dataset_scale.value,
        }

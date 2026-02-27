"""
Optuna objective for DSLFM/PC-only optimization.

All legacy ensemble paths were removed; the objective now samples only
DSLFM hyperparameters and scores using the composite HPO metric, which
prioritizes ranking (MRR/Hits) and then classification metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pff.domain.hpo.bounds import get_range
from pff.domain.hpo.models import KGE_MODEL_DSLFM
from pff.domain.hpo.search_space import SearchSpaceFactory
from pff.infrastructure.hpo.config_loader import (
    load_adaptive_range_factors,
    load_metric_bounds,
)
from pff.shared.core.file_manager import FileManager
from pff.shared.system.cuda import is_cuda_available

from .artifacts import TrialArtifactManager
from .pipeline import evaluate_trial


@dataclass
class ObjectiveContext:
    """Context for the DSLFM Optuna objective."""

    trial_runs_dir: Path
    hpo_ranges: dict[str, Any]
    file_manager: FileManager
    artifact_manager: TrialArtifactManager


def _infer_dataset_stats(
    train_df: pl.DataFrame, valid_df: pl.DataFrame | None
) -> tuple[int, int]:
    """Infer entity/relation counts from Parquet splits."""
    cols = train_df.columns
    if {"s", "p", "o"}.issubset(set(cols)):
        h_col, r_col, t_col = "s", "p", "o"
    elif {"head", "relation", "tail"}.issubset(set(cols)):
        h_col, r_col, t_col = "head", "relation", "tail"
    else:
        h_col, r_col, t_col = cols[0], cols[1], cols[2]

    frames = [train_df]
    if valid_df is not None:
        frames.append(valid_df)
    combined = pl.concat(frames)
    ent_series = pl.concat([combined[h_col], combined[t_col]])
    rel_series = combined[r_col]
    num_entities_unique = int(ent_series.n_unique())
    num_relations_unique = int(rel_series.n_unique())

    entity_upper_bound = -1
    relation_upper_bound = -1
    if (
        ent_series.dtype.is_integer()
        and ent_series.null_count() == 0
        and len(ent_series) > 0
    ):
        max_entity = ent_series.max()
        if max_entity is not None:
            entity_upper_bound = int(max_entity)
    if (
        rel_series.dtype.is_integer()
        and rel_series.null_count() == 0
        and len(rel_series) > 0
    ):
        max_relation = rel_series.max()
        if max_relation is not None:
            relation_upper_bound = int(max_relation)

    num_entities = max(
        num_entities_unique,
        entity_upper_bound + 1 if entity_upper_bound >= 0 else 0,
    )
    num_relations = max(
        num_relations_unique,
        relation_upper_bound + 1 if relation_upper_bound >= 0 else 0,
    )
    return int(num_entities), int(num_relations)


def _suggest_dslfm_params(
    trial,
    hpo_ranges: dict[str, Any],
    *,
    num_train: int,
    num_valid: int,
    num_entities: int,
    num_relations: int,
    adaptive_bounds: dict[str, Any],
) -> dict[str, Any]:
    """Sample DSLFM hyperparameters from configuration bounds.

    Bounds come from `config/hpo/optimization.yaml` and are optionally adjusted
    using dataset-aware `adaptive_bounds`. For CPU-only environments, the upper
    ranges are clamped to keep trials responsive while preserving MRR-sensitive
    knobs such as negative sampling.
    """
    has_cuda = is_cuda_available()

    kge_bounds = hpo_ranges.get("kge", {})
    training_bounds = hpo_ranges.get("training", {})
    logic_bounds = hpo_ranges.get("logic", {})
    pc_bounds = hpo_ranges.get("pc", {})
    regularization_bounds = hpo_ranges.get("regularization", {})
    contrastive_bounds = hpo_ranges.get("contrastive", {})
    architecture_bounds = hpo_ranges.get("architecture", {})
    metric_bounds = load_metric_bounds()

    def _require_range(bounds: dict[str, Any], key: str) -> tuple[float, float]:
        entry = bounds.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing bounds for {key} in HPO ranges.")
        low = entry.get("low")
        high = entry.get("high")
        if low is None or high is None:
            raise ValueError(f"Missing low/high for {key} in HPO ranges.")
        low_f, high_f = float(low), float(high)
        if low_f > high_f:
            raise ValueError(f"Invalid range for {key}: low={low_f} high={high_f}.")
        return low_f, high_f

    def _require_choices(bounds: dict[str, Any], key: str) -> list[Any]:
        entry = bounds.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing choices for {key} in HPO ranges.")
        choices = entry.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Empty choices for {key} in HPO ranges.")
        return choices

    def _require_value(bounds: dict[str, Any], key: str) -> Any:
        if key not in bounds:
            raise ValueError(f"Missing {key} in HPO ranges.")
        return bounds[key]

    batch_low, batch_high = _require_range(kge_bounds, "batch_size")
    neg_low, neg_high = _require_range(kge_bounds, "negative_sample_size")
    adv_low, adv_high = _require_range(kge_bounds, "adversarial_temperature")
    lr_low, lr_high = _require_range(kge_bounds, "learning_rate")
    contrastive_temp_low = float(_require_value(contrastive_bounds, "temperature_low"))
    contrastive_temp_high = float(
        _require_value(contrastive_bounds, "temperature_high")
    )
    num_global_neg_low = int(
        _require_value(contrastive_bounds, "num_global_negatives_low")
    )
    num_global_neg_high = int(
        _require_value(contrastive_bounds, "num_global_negatives_high")
    )
    kl_weight_low = float(_require_value(architecture_bounds, "kl_weight_low"))
    kl_weight_high = float(_require_value(architecture_bounds, "kl_weight_high"))

    embedding_choices = [
        int(choice) for choice in _require_choices(kge_bounds, "embedding_dim")
    ]
    max_communities_choices = [
        int(choice) for choice in _require_choices(kge_bounds, "max_communities")
    ]
    self_adv_choices = [
        bool(choice) for choice in _require_choices(kge_bounds, "self_adversarial")
    ]
    use_bert_default = bool(_require_value(kge_bounds, "use_bert_default"))
    t_norm_choices = [
        str(choice) for choice in _require_choices(logic_bounds, "t_norm")
    ]
    attr_hidden_choices = [
        int(choice) for choice in _require_choices(logic_bounds, "attr_hidden_dim")
    ]
    depth_choices = [
        int(choice) for choice in _require_choices(pc_bounds, "max_circuit_depth")
    ]

    lambda_logic_low, lambda_logic_high = _require_range(logic_bounds, "lambda_logic")
    lambda_pc_low, lambda_pc_high = _require_range(pc_bounds, "lambda_pc")
    ibp_alpha_low, ibp_alpha_high = _require_range(kge_bounds, "ibp_alpha")
    prune_low, prune_high = _require_range(pc_bounds, "pruning_threshold")
    rebuild_low, rebuild_high = _require_range(pc_bounds, "rebuild_every")
    lambda_sum_cap = max(
        0.0, float(_require_value(regularization_bounds, "lambda_sum_cap"))
    )

    if not has_cuda:
        batch_high = min(batch_high, 512)
        lambda_pc_high = min(lambda_pc_high, 0.02)

    def _cap_int_range(
        low_raw: float,
        high_raw: float,
        *,
        cap_high: int,
        floor_low: int,
    ) -> tuple[int, int]:
        """Clamp an integer range to a maximum, ensuring a valid Optuna bound."""
        low = int(low_raw)
        high = int(high_raw)
        high = min(high, cap_high)
        low = min(low, high)
        if low >= high:
            low = max(floor_low, int(0.6 * high))
            low = min(low, high)
        return low, high

    def _align_step_range(
        low_raw: float, high_raw: float, step: int
    ) -> tuple[int, int]:
        low = int(low_raw)
        high = int(high_raw)
        if step <= 0:
            return low, high
        if low > high:
            return low, low
        span = high - low
        aligned_high = low + (span // step) * step
        return low, max(low, aligned_high)

    neg_low, neg_high = _align_step_range(neg_low, neg_high, 64)

    epochs_low, epochs_high = _cap_int_range(
        adaptive_bounds["epochs"][0],
        adaptive_bounds["epochs"][1],
        cap_high=120 if not has_cuda else int(adaptive_bounds["epochs"][1]),
        floor_low=8,
    )
    if isinstance(training_bounds, dict) and training_bounds.get("epochs"):
        override_low, override_high = get_range(
            training_bounds,
            ["epochs"],
            epochs_low,
            epochs_high,
        )
        epochs_low, epochs_high = _cap_int_range(
            override_low,
            override_high,
            cap_high=int(override_high),
            floor_low=8,
        )

    patience_low, patience_high = _cap_int_range(
        adaptive_bounds["early_stopping_patience"][0],
        adaptive_bounds["early_stopping_patience"][1],
        cap_high=(
            25 if not has_cuda else int(adaptive_bounds["early_stopping_patience"][1])
        ),
        floor_low=5,
    )

    adaptive_batch_low = int(
        adaptive_bounds.get("batch_size", (batch_low, batch_high))[0]
    )
    adaptive_batch_high = int(
        adaptive_bounds.get("batch_size", (batch_low, batch_high))[1]
    )
    if batch_low == batch_high:
        resolved_batch_low = int(batch_low)
        resolved_batch_high = int(batch_high)
    else:
        resolved_batch_low = max(int(batch_low), adaptive_batch_low)
        resolved_batch_high = min(int(batch_high), adaptive_batch_high)
        if resolved_batch_low > resolved_batch_high:
            resolved_batch_low = int(batch_low)
            resolved_batch_high = int(batch_high)
    training_use_compile = (
        bool(training_bounds.get("use_compile", False))
        if isinstance(training_bounds, dict)
        else False
    )

    params = {
        "kge_model": KGE_MODEL_DSLFM,
        "embedding_dim": trial.suggest_categorical("embedding_dim", embedding_choices),
        "max_communities": trial.suggest_categorical(
            "max_communities", max_communities_choices
        ),
        "ibp_alpha": trial.suggest_float("ibp_alpha", ibp_alpha_low, ibp_alpha_high),
        "dslfm_epochs": trial.suggest_int(
            "dslfm_epochs",
            epochs_low,
            epochs_high,
        ),
        "early_stopping_patience": trial.suggest_int(
            "early_stopping_patience",
            patience_low,
            patience_high,
        ),
        "batch_size": (
            resolved_batch_low
            if resolved_batch_low == resolved_batch_high
            else trial.suggest_int(
                "batch_size", resolved_batch_low, resolved_batch_high
            )
        ),
        "negative_sample_size": trial.suggest_int(
            "negative_sample_size",
            neg_low,
            neg_high,
            step=64,
        ),
        "adversarial_temperature": trial.suggest_float(
            "adversarial_temperature", adv_low, adv_high
        ),
        "self_adversarial": (
            self_adv_choices[0]
            if len(self_adv_choices) <= 1
            else trial.suggest_categorical("self_adversarial", self_adv_choices)
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", lr_low, lr_high, log=True
        ),
        "lambda_logic": trial.suggest_float(
            "lambda_logic", lambda_logic_low, lambda_logic_high
        ),
        "t_norm": trial.suggest_categorical("t_norm", t_norm_choices),
        "attr_hidden_dim": trial.suggest_categorical(
            "attr_hidden_dim", attr_hidden_choices
        ),
        "lambda_pc": trial.suggest_float("lambda_pc", lambda_pc_low, lambda_pc_high),
        "pruning_threshold": trial.suggest_float(
            "pruning_threshold", prune_low, prune_high, log=True
        ),
        "rebuild_every": trial.suggest_int(
            "rebuild_every", int(rebuild_low), int(rebuild_high), step=5
        ),
        "max_circuit_depth": trial.suggest_categorical(
            "max_circuit_depth", depth_choices
        ),
        "min_delta": trial.suggest_float(
            "min_delta",
            min(1e-5, float(adaptive_bounds["min_delta"][0])),
            float(adaptive_bounds["min_delta"][1]),
        ),
        "validate_every": trial.suggest_int(
            "validate_every",
            int(adaptive_bounds["validate_every"][0]),
            int(adaptive_bounds["validate_every"][1]),
        ),
        "contrastive_temperature": trial.suggest_float(
            "contrastive_temperature",
            contrastive_temp_low,
            contrastive_temp_high,
        ),
        "num_global_negatives": trial.suggest_int(
            "num_global_negatives",
            num_global_neg_low,
            num_global_neg_high,
            step=32,
        ),
        "kl_weight": trial.suggest_float(
            "kl_weight",
            kl_weight_low,
            kl_weight_high,
            log=True,
        ),
        # Keep this compile switch config-driven via optimization.yaml to avoid hardcoded behavior across environments.
        "use_compile": training_use_compile,
        # Keep validation latents fresh by default; stale entity caches can hide ranking gains.
        "refresh_cache_on_val": True,
        "use_bert": use_bert_default,
    }

    params["lambda_logic"] = min(params["lambda_logic"], lambda_sum_cap)
    params["lambda_pc"] = min(
        params["lambda_pc"], max(0.0, lambda_sum_cap - params["lambda_logic"])
    )

    params["batch_size"] = min(params["batch_size"], int(batch_high))
    params["negative_sample_size"] = min(params["negative_sample_size"], int(neg_high))
    params["metric_bounds"] = metric_bounds
    return params


def collect_dslfm_distributions(
    hpo_ranges: dict[str, Any],
    *,
    num_train: int,
    num_valid: int,
    num_entities: int,
    num_relations: int,
    adaptive_bounds: dict[str, Any],
) -> dict[str, Any]:
    """Collect Optuna distributions for the DSLFM search space (warm-start validation)."""
    try:
        from optuna.distributions import (
            CategoricalDistribution,
            FloatDistribution,
            IntDistribution,
        )
    except Exception:
        return {}

    class _DistributionCollector:
        def __init__(self) -> None:
            """Execute init."""

            self.distributions: dict[str, Any] = {}

        def suggest_int(
            self,
            name: str,
            low: int,
            high: int,
            *,
            step: int | None = None,
            log: bool = False,
        ) -> int:
            """Execute suggest int.



            Args:

                name: Input value used by this callable.

                low: Input value used by this callable.

                high: Input value used by this callable.

                step: Optional input value.

                log: Optional input value.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            self.distributions[name] = IntDistribution(
                low=int(low),
                high=int(high),
                log=bool(log),
                step=int(step) if step else 1,
            )
            return int(low)

        def suggest_float(
            self,
            name: str,
            low: float,
            high: float,
            *,
            step: float | None = None,
            log: bool = False,
        ) -> float:
            """Execute suggest float.



            Args:

                name: Input value used by this callable.

                low: Input value used by this callable.

                high: Input value used by this callable.

                step: Optional input value.

                log: Optional input value.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            self.distributions[name] = FloatDistribution(
                low=float(low),
                high=float(high),
                log=bool(log),
                step=step,
            )
            return float(low)

        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
            """Execute suggest categorical.



            Args:

                name: Input value used by this callable.

                choices: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

            self.distributions[name] = CategoricalDistribution(choices=choices)
            return choices[0] if choices else None

    collector = _DistributionCollector()
    _suggest_dslfm_params(
        collector,
        hpo_ranges,
        num_train=num_train,
        num_valid=num_valid,
        num_entities=num_entities,
        num_relations=num_relations,
        adaptive_bounds=adaptive_bounds,
    )
    return collector.distributions


def kg_objective(
    trial,
    *,
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    target_entity_ratio: float,
    trial_runs_dir: Path,
    hpo_ranges: dict[str, dict[str, int | float]],
    file_manager: FileManager,
    artifact_manager: TrialArtifactManager,
    precomputed_stats: tuple[int, int] | None = None,
    precomputed_adaptive_bounds: dict[str, Any] | None = None,
) -> float:
    """Optuna objective for DSLFM/PC-only HPO.

    Returns:
        float: Trial score (multi-metric composite).
    """
    if trial_runs_dir is None:
        raise RuntimeError("Trial output directory not initialized")

    if precomputed_stats is not None:
        num_entities, num_relations = precomputed_stats
    else:
        num_entities, num_relations = _infer_dataset_stats(train_df, valid_df)

    if precomputed_adaptive_bounds is not None:
        adaptive_bounds = precomputed_adaptive_bounds
    else:
        range_factors = load_adaptive_range_factors(file_manager)
        adaptive_bounds = SearchSpaceFactory.create_adaptive_training_space(
            num_train_triples=len(train_df),
            num_valid_triples=len(valid_df),
            num_entities=num_entities,
            num_relations=num_relations,
            range_factors=range_factors,
        )
    params = _suggest_dslfm_params(
        trial,
        hpo_ranges,
        num_train=len(train_df),
        num_valid=len(valid_df),
        num_entities=num_entities,
        num_relations=num_relations,
        adaptive_bounds=adaptive_bounds,
    )

    score = evaluate_trial(
        params=params,
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=target_entity_ratio,
        trial_number=trial.number,
        trial_output_root=trial_runs_dir,
        trial=trial,
        artifact_manager=artifact_manager,
    )
    return score

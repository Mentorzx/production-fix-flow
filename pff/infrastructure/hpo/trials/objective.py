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
    num_entities = ent_series.n_unique()
    num_relations = rel_series.n_unique()
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

    batch_low, batch_high = get_range(kge_bounds, ["batch_size"], 192, 512)
    neg_low, neg_high = get_range(kge_bounds, ["negative_sample_size"], 256, 512)
    adv_low, adv_high = get_range(kge_bounds, ["adversarial_temperature"], 0.5, 5.0)
    lr_low, lr_high = get_range(kge_bounds, ["learning_rate"], 1e-5, 1e-3)
    contrastive_temp_low = float(contrastive_bounds.get("temperature_low", 0.05))
    contrastive_temp_high = float(contrastive_bounds.get("temperature_high", 0.2))
    num_global_neg_low = int(contrastive_bounds.get("num_global_negatives_low", 64))
    num_global_neg_high = int(contrastive_bounds.get("num_global_negatives_high", 256))
    kl_weight_low = float(architecture_bounds.get("kl_weight_low", 1e-4))
    kl_weight_high = float(architecture_bounds.get("kl_weight_high", 5e-2))

    raw_embedding_choices = kge_bounds.get("embedding_dim", {}).get(
        "choices", [128, 256]
    )
    embedding_choices = (
        [int(choice) for choice in raw_embedding_choices]
        if raw_embedding_choices
        else [128, 256]
    )
    raw_max_communities = kge_bounds.get("max_communities", {}).get("choices", [128])
    max_communities_choices = (
        [int(choice) for choice in raw_max_communities]
        if raw_max_communities
        else [128]
    )
    raw_self_adv_choices = kge_bounds.get("self_adversarial", {}).get(
        "choices", [False]
    )
    self_adv_choices = (
        [bool(choice) for choice in raw_self_adv_choices]
        if raw_self_adv_choices
        else [False]
    )
    use_bert_default = bool(kge_bounds.get("use_bert_default", False))
    raw_t_norm_choices = logic_bounds.get("t_norm", {}).get(
        "choices", ["product", "lukasiewicz"]
    )
    t_norm_choices = (
        list(raw_t_norm_choices) if raw_t_norm_choices else ["product", "lukasiewicz"]
    )
    raw_attr_hidden_choices = logic_bounds.get("attr_hidden_dim", {}).get(
        "choices", [64, 128, 256]
    )
    attr_hidden_choices = (
        [int(choice) for choice in raw_attr_hidden_choices]
        if raw_attr_hidden_choices
        else [64, 128, 256]
    )
    raw_depth_choices = pc_bounds.get("max_circuit_depth", {}).get(
        "choices", [2, 3, 4, 5, 6, 7, 8]
    )
    depth_choices = (
        [int(choice) for choice in raw_depth_choices]
        if raw_depth_choices
        else [2, 3, 4, 5, 6, 7, 8]
    )

    lambda_logic_low, lambda_logic_high = get_range(
        logic_bounds, ["lambda_logic"], 0.0, 0.6
    )
    lambda_pc_low, lambda_pc_high = get_range(pc_bounds, ["lambda_pc"], 0.0, 0.6)
    ibp_alpha_low, ibp_alpha_high = get_range(kge_bounds, ["ibp_alpha"], 1.0, 10.0)
    prune_low, prune_high = get_range(pc_bounds, ["pruning_threshold"], 1e-3, 1e-1)
    rebuild_low, rebuild_high = get_range(pc_bounds, ["rebuild_every"], 0, 50)
    lambda_sum_cap = max(0.0, float(regularization_bounds.get("lambda_sum_cap", 0.7)))

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

    epochs_low, epochs_high = _cap_int_range(
        adaptive_bounds["epochs"][0],
        adaptive_bounds["epochs"][1],
        cap_high=120 if not has_cuda else int(adaptive_bounds["epochs"][1]),
        floor_low=8,
    )

    epochs_low = min(epochs_low, 50)
    epochs_high = max(epochs_high, 200)
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

    patience_low = min(patience_low, 5)
    patience_high = max(patience_high, 25)

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
        "refresh_cache_on_val": False,
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
            self.distributions[name] = FloatDistribution(
                low=float(low),
                high=float(high),
                log=bool(log),
                step=step,
            )
            return float(low)

        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
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

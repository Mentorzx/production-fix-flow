"""
DSLFM-KGC Trial Evaluation Pipeline.

Trains DSLFM-KGC (BERT + VAE + IBP + Probabilistic Circuits) and computes
multi-metric composite scores across ranking, classification and efficiency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl

from pff.domain.hpo.models import KGE_MODEL_DSLFM
from pff.domain.hpo.scoring import (
    build_weights_from_settings,
    compute_score,
    rename_metric_keys,
)
from pff.shared import logger
from pff.shared.acceleration.concurrency import ConcurrencyManager
from pff.shared.core.file_manager import FileManager
from pff.shared.determinism import set_global_seed
from pff.shared.hash import stable_hash
from pff.shared.ops.global_interrupt_manager import check_interruption
from pff.shared.system.cuda import is_cuda_available
from pff.shared.system.resource_manager import get_memory_safe_workers

from .artifacts import TrialArtifactManager
from .config_loader import get_cached_config, load_scoring_settings
from .evaluator import _train_dslfm_kgc_model


@dataclass
class TrialEvaluationConfig:
    """Parameter object for DSLFM trial evaluation."""

    params: dict[str, Any]
    train_df: pl.DataFrame
    valid_df: pl.DataFrame
    target_entity_ratio: float
    trial_number: int
    trial_output_root: Path
    trial: Any | None = None
    artifact_manager: TrialArtifactManager | None = field(default=None)

    def __post_init__(self) -> None:
        if self.artifact_manager is None:
            self.artifact_manager = TrialArtifactManager(base_dir=None)
        if getattr(self.artifact_manager, "store", None) is None:
            raise ValueError("HPO trial artifacts require a Postgres store")


class TrialEvaluationPipeline:
    """Minimal DSLFM-only trial pipeline."""

    def __init__(
        self,
        params: dict[str, Any],
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame,
        target_entity_ratio: float,
        trial_number: int,
        trial_output_root: Path,
        *,
        trial: Any | None = None,
        artifact_manager: TrialArtifactManager | None = None,
        enable_cross_validation: bool = True,
        cv_fold_id: int | None = None,
    ) -> None:
        self.params = params
        self.train_df = train_df
        self.valid_df = valid_df
        self.target_entity_ratio = target_entity_ratio
        self.trial_number = trial_number
        self.trial_output_root = trial_output_root
        self.trial = trial
        self.artifact_manager = artifact_manager or TrialArtifactManager(base_dir=None)
        if getattr(self.artifact_manager, "store", None) is None:
            raise ValueError("HPO trial artifacts require a Postgres store")
        self.file_manager = FileManager()
        self.enable_cross_validation = enable_cross_validation
        self.cv_fold_id = cv_fold_id
        self.cv_settings = self._load_cv_settings()

        self.trial_dir: Path | None = None
        self.config_dir: Path | None = None
        self.models_dir: Path | None = None
        self.kge_model_dir: Path | None = None
        self.kge_checkpoint_path: Path | None = None

        self.kge_metrics: dict[str, float] = {}
        self.normalized_metrics: dict[str, float] = {}
        self.score_components: dict[str, float] = {}
        self.elapsed_time: float = 0.0
        self.composite_score: float = 0.0
        self.base_score: float = 0.0
        self.model_paths: dict[str, str] = {}

    def _update_live_status_preparing(self) -> None:
        """Update dashboard status to show trial is preparing."""
        try:
            from datetime import datetime, timezone

            from pff.shared.core.config import settings

            trial_attrs = getattr(self.trial, "user_attrs", {}) or {}
            warmstart = bool(
                trial_attrs.get("warmstart") or trial_attrs.get("warmstart_seed")
            )

            status_path = (
                settings.OUTPUTS_DIR / "optimization" / "plots" / "live_status.json"
            )

            status = {
                "trial_number": self.trial_number,
                "cv_fold_id": self.cv_fold_id,
                "params": self.params,
                "warmstart": warmstart,
                "current_epoch": 0,
                "total_epochs": int(self.params.get("dslfm_epochs", 0)),
                "elapsed_seconds": 0.0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "epoch_history": [],
                "recent_logs": [],
                "progress": 0.0,
                "status": "PREPARING",
            }

            status_path.parent.mkdir(parents=True, exist_ok=True)

            FileManager().save(status, status_path)

        except Exception as e:
            logger.debug(f"Failed to update live status (PREPARING): {e}")

    def run(self) -> float:
        """Execute DSLFM training and compute the composite score.

        Returns:
            float: Composite score (ranking-first, then classification and efficiency).
        """
        check_interruption()
        self._update_live_status_preparing()
        start = time.time()
        self._setup_trial()
        check_interruption()
        if self.enable_cross_validation and self.cv_settings["cv_folds"] > 1:
            self.composite_score = self._run_cross_validation()
            self.elapsed_time = time.time() - start
            self._record_result()
            return self.composite_score

        self._train_kge()
        check_interruption()
        self.elapsed_time = time.time() - start
        self._compute_score()
        self._record_result()
        logger.info(f"Tempo total do trial: {self.elapsed_time:.2f}s")
        return self.composite_score

    def _load_cv_settings(self) -> dict[str, Any]:
        defaults = {"cv_folds": 1, "cv_parallel": False, "cv_max_workers": 2}
        cfg = get_cached_config("config/hpo/optimization.yaml", FileManager())
        defaults_cfg = cfg.get("defaults", {}) if isinstance(cfg, dict) else {}
        return {
            "cv_folds": int(defaults_cfg.get("cv_folds", defaults["cv_folds"])),
            "cv_parallel": bool(
                defaults_cfg.get("cv_parallel", defaults["cv_parallel"])
            ),
            "cv_max_workers": int(
                defaults_cfg.get("cv_max_workers", defaults["cv_max_workers"])
            ),
        }

    def _run_cross_validation(self) -> float:
        cv_folds = self.cv_settings["cv_folds"]
        cv_parallel = self._resolve_cv_parallel(self.cv_settings["cv_parallel"])
        cv_workers = self.cv_settings["cv_max_workers"]
        logger.info(
            f"Iniciando cross-validation: folds={cv_folds} paralelo={cv_parallel}"
        )

        rng = np.random.default_rng(self.trial_seed)
        indices = np.arange(len(self.train_df))
        rng.shuffle(indices)
        fold_indices = np.array_split(indices, cv_folds)

        def _run_fold(
            fold_id: int,
            val_idx: np.ndarray,
        ) -> tuple[float, dict[str, float], float]:
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
            train_df = self.train_df[train_idx]
            valid_df = self.train_df[val_idx]
            fold_root = self.trial_output_root / "cv_folds" / f"fold_{fold_id:02d}"
            fold_pipeline = TrialEvaluationPipeline(
                params=self.params,
                train_df=train_df,
                valid_df=valid_df,
                target_entity_ratio=self.target_entity_ratio,
                trial_number=self.trial_number,
                trial_output_root=fold_root,
                trial=None,
                artifact_manager=TrialArtifactManager(
                    base_dir=(
                        None
                        if getattr(self.artifact_manager, "store", None) is not None
                        else fold_root / "results"
                    ),
                    study_name=getattr(self.artifact_manager, "study_name", None),
                    store=getattr(self.artifact_manager, "store", None),
                ),
                enable_cross_validation=False,
                cv_fold_id=fold_id,
            )
            score = float(fold_pipeline.run())
            metrics = dict(getattr(fold_pipeline, "kge_metrics", {}) or {})
            elapsed = float(getattr(fold_pipeline, "elapsed_time", 0.0) or 0.0)
            return score, metrics, elapsed

        fold_args = [(idx, fold_indices[idx]) for idx in range(cv_folds)]
        if cv_parallel:
            concurrency = ConcurrencyManager()
            results = concurrency.execute_sync(
                _run_fold,
                fold_args,
                task_type="io_thread",
                max_workers=cv_workers,
                desc="cv_folds",
            )
        else:
            results = []
            for idx in range(cv_folds):
                result = _run_fold(idx, fold_indices[idx])
                results.append(result)
                self._cleanup_after_fold()

        scores: list[float] = []
        metric_sums: dict[str, list[float]] = {}
        for score, metrics, _elapsed in results:
            scores.append(float(score))
            for key, value in metrics.items():
                try:
                    metric_sums.setdefault(key, []).append(float(value))
                except Exception:
                    continue

        self.kge_metrics = {
            key: float(np.mean(values)) for key, values in metric_sums.items() if values
        }
        mean_score = float(np.mean(scores)) if scores else 0.0
        logger.info(f"Cross-validation concluida: score_medio={mean_score:.4f}")
        return mean_score

    def _resolve_cv_parallel(self, requested_parallel: bool) -> bool:
        if not requested_parallel:
            return False

        if is_cuda_available():
            logger.debug(
                "Parallel cross-validation disabled: CUDA detected; "
                "single-GPU folds should run sequentially."
            )
            return False

        try:
            requested_workers = int(self.params.get("num_workers", 0))
        except (TypeError, ValueError):
            requested_workers = 0

        if requested_workers > 0:
            logger.debug(
                "Parallel cross-validation disabled: DataLoader workers enabled; "
                "threaded CV can deadlock."
            )
            return False

        try:
            batch_size = int(self.params.get("batch_size", 1000))
        except (TypeError, ValueError):
            batch_size = 1000

        auto_workers = get_memory_safe_workers(chunk_size=batch_size)
        if auto_workers > 0:
            logger.debug(
                "Parallel cross-validation disabled: auto DataLoader workers would "
                "spawn processes under threaded CV."
            )
            return False

        return True

    def _cleanup_after_fold(self) -> None:
        """Clean up GPU memory after completing a CV fold."""
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def _setup_trial(self) -> None:
        """Prepare trial directories and deterministic seed."""
        check_interruption()
        dataset_msg = (
            f"Visao do dataset do trial: treino={len(self.train_df):,} | "
            f"validacao={len(self.valid_df):,}"
        )
        if self.cv_fold_id is None:
            logger.info(dataset_msg)
        else:
            logger.debug(f"{dataset_msg} fold={self.cv_fold_id:02d}")
        self.trial_seed = stable_hash(
            tuple(sorted(self.params.items())), truncate=16
        ) & (2**32 - 1)

        set_global_seed(self.trial_seed)
        logger.debug(f"trial_seed={self.trial_seed} applied (deterministic mode)")

        if self.trial is not None:
            try:
                self.trial.set_user_attr("trial_seed", self.trial_seed)
            except Exception as exc:
                logger.debug(
                    f"Failed to set Optuna trial user attribute trial_seed: {exc}"
                )

        self.trial_dir = self.trial_output_root / f"trial_{self.trial_number:04d}"
        self.file_manager.delete_directory(self.trial_dir, ignore_errors=True)
        self.file_manager.ensure_dir(self.trial_dir)

        self.config_dir = self.trial_dir / "config"
        self.file_manager.ensure_dir(self.config_dir)
        self.models_dir = self.trial_dir / "models"
        self.file_manager.ensure_dir(self.models_dir)
        self.kge_model_dir = self.models_dir / KGE_MODEL_DSLFM
        self.file_manager.ensure_dir(self.kge_model_dir)

    def _train_kge(self) -> None:
        """Train DSLFM-KGC model and collect metrics."""
        import numpy as np

        check_interruption()
        start = time.time()
        if self.kge_model_dir is None:
            raise RuntimeError("Model directory not initialized")
        kge_checkpoint_dir = self.kge_model_dir / "checkpoints"
        self.file_manager.ensure_dir(kge_checkpoint_dir)

        entity_ids = pl.concat(
            [
                self.train_df["s"],
                self.train_df["o"],
                self.valid_df["s"],
                self.valid_df["o"],
            ]
        )
        relation_ids = pl.concat([self.train_df["p"], self.valid_df["p"]])

        entity_is_contiguous = False
        relation_is_contiguous = False
        num_entities = 0
        num_relations = 0

        if entity_ids.null_count() == 0 and entity_ids.dtype.is_integer():
            entity_min = entity_ids.min()
            entity_max = entity_ids.max()
            if entity_min == 0 and entity_max is not None:
                unique_entities = int(entity_ids.n_unique())

                if int(entity_max) == unique_entities - 1:
                    entity_is_contiguous = True
                    num_entities = int(entity_max) + 1

        if relation_ids.null_count() == 0 and relation_ids.dtype.is_integer():
            rel_min = relation_ids.min()
            rel_max = relation_ids.max()
            if rel_min == 0 and rel_max is not None:
                unique_relations = int(relation_ids.n_unique())
                if int(rel_max) == unique_relations - 1:
                    relation_is_contiguous = True
                    num_relations = int(rel_max) + 1

        if entity_is_contiguous and relation_is_contiguous:
            train_triples = np.asarray(
                self.train_df.select(["s", "p", "o"]).to_numpy(),
                dtype=np.int64,
            )
            valid_triples = np.asarray(
                self.valid_df.select(["s", "p", "o"]).to_numpy(),
                dtype=np.int64,
            )
            relation_names = [str(i) for i in range(num_relations)]
        else:
            entity_labels = (
                pl.concat(
                    [
                        self.train_df["s"],
                        self.train_df["o"],
                        self.valid_df["s"],
                        self.valid_df["o"],
                    ]
                )
                .unique()
                .sort()
            )
            relation_labels = (
                pl.concat([self.train_df["p"], self.valid_df["p"]]).unique().sort()
            )

            entity_map = pl.DataFrame({"label": entity_labels}).with_row_index("id")
            relation_map = pl.DataFrame({"label": relation_labels}).with_row_index("id")
            relation_names = relation_map["label"].to_list()
            combined = pl.concat(
                [
                    self.train_df.with_columns(pl.lit("train").alias("__split")),
                    self.valid_df.with_columns(pl.lit("valid").alias("__split")),
                ],
            )
            mapped = (
                combined.select(["s", "p", "o", "__split"])
                .join(
                    entity_map,
                    left_on="s",
                    right_on="label",
                    how="left",
                    maintain_order="left",
                )
                .rename({"id": "s_id"})
                .join(
                    relation_map,
                    left_on="p",
                    right_on="label",
                    how="left",
                    maintain_order="left",
                )
                .rename({"id": "p_id"})
                .join(
                    entity_map,
                    left_on="o",
                    right_on="label",
                    how="left",
                    maintain_order="left",
                )
                .rename({"id": "o_id"})
                .select(["__split", "s_id", "p_id", "o_id"])
            )
            train_triples = np.asarray(
                mapped.filter(pl.col("__split") == "train")
                .select(["s_id", "p_id", "o_id"])
                .to_numpy(),
                dtype=np.int64,
            )
            valid_triples = np.asarray(
                mapped.filter(pl.col("__split") == "valid")
                .select(["s_id", "p_id", "o_id"])
                .to_numpy(),
                dtype=np.int64,
            )
            num_entities = int(entity_map.height)
            num_relations = int(relation_map.height)

        logger.info(
            f"DSLFM-KGC: entidades={num_entities:,}, relacoes={num_relations}, "
            f"train={len(train_triples):,}, valid={len(valid_triples):,}"
        )

        try:
            kge_stats, checkpoint_path = _train_dslfm_kgc_model(
                params=self.params,
                model_dir=self.kge_model_dir,
                train_triples=train_triples,
                valid_triples=valid_triples,
                num_entities=num_entities,
                num_relations=num_relations,
                relation_names=relation_names,
                use_bert=self.params.get("use_bert", True),
                trial=self.trial,
                trial_number_override=self.trial_number,
                cv_fold_id=self.cv_fold_id,
            )
        except optuna.TrialPruned:
            logger.info(
                "Trial pruned by Optuna", stop_reason="pruning", params=self.params
            )
            self.elapsed_time = time.time() - start
            raise
        except Exception as e:
            logger.error(
                "Training failed",
                error=str(e),
                trial_number=self.trial_number,
                params=self.params,
            )
            raise

        best_metrics = kge_stats.get("best_metrics", {})
        raw_metrics = kge_stats.get("final_metrics", {})

        if best_metrics:
            for k, v in best_metrics.items():
                if k not in raw_metrics or raw_metrics[k] == 0:
                    raw_metrics[k] = v

        raw_metrics["best_mrr"] = kge_stats.get(
            "best_val_mrr", raw_metrics.get("mrr", 0.0)
        )
        best_mcc = kge_stats.get("best_val_mcc", raw_metrics.get("mcc", 0.0))
        raw_metrics["best_mcc"] = best_mcc
        if best_mcc > raw_metrics.get("mcc", 0.0):
            raw_metrics["mcc"] = best_mcc
        self.kge_metrics = rename_metric_keys(raw_metrics)
        self.kge_checkpoint_path = checkpoint_path
        self.model_paths["dslfm"] = str(checkpoint_path)

    def _compute_score(self) -> None:
        """Compute composite score using all metrics with min-max normalization."""
        check_interruption()
        scoring_settings = load_scoring_settings(self.file_manager)
        weights = build_weights_from_settings(scoring_settings)
        metrics_for_score = rename_metric_keys(
            {**self.kge_metrics, "duration": self.elapsed_time}
        )
        history_metrics = self.artifact_manager.list_metrics()
        score, normalized, components = compute_score(
            metrics_for_score, history_metrics, weights=weights
        )
        self.base_score = score

        self.composite_score = float(self.base_score)
        self.normalized_metrics = normalized
        self.score_components = {
            "rank": components.rank,
            "classification": components.classification,
            "efficiency": components.efficiency,
        }
        logger.success(
            "Avaliacao do trial DSLFM concluida: "
            f"score={self.composite_score:.4f} "
            f"(rank={components.rank:.4f}, clf={components.classification:.4f}, tempo={components.efficiency:.4f})"
        )

        fold_suffix = (
            f" (Fold {self.cv_fold_id})" if self.cv_fold_id is not None else ""
        )
        logger.info(
            f"Resumo do trial #{self.trial_number + 1}{fold_suffix}: score={self.composite_score:.4f}, "
            f"duracao={self.elapsed_time:.2f}s"
        )

    def _record_result(self) -> None:
        """Persist trial artifacts and metrics."""
        if self.trial_dir is None:
            return
        metrics_with_duration = {**self.kge_metrics, "duration": self.elapsed_time}
        model_metrics = {
            "metrics": self.kge_metrics,
            "normalized_metrics": self.normalized_metrics,
            "composite_score": self.composite_score,
            "score_components": self.score_components,
            "elapsed_time": self.elapsed_time,
        }
        model_paths = {k: str(v) for k, v in self.model_paths.items()}
        trial_result = {
            "metrics": metrics_with_duration,
            "normalized_metrics": self.normalized_metrics,
            "model_metrics": model_metrics,
            "params": dict(self.params),
            "trial_number": self.trial_number,
            "trial_dir": str(self.trial_dir),
            "model_paths": model_paths,
            "models_trained": {
                "dslfm": bool(
                    self.kge_checkpoint_path
                    and self.file_manager.exists(self.kge_checkpoint_path)
                )
            },
            "elapsed_time": self.elapsed_time,
            "score": self.composite_score,
        }
        self.artifact_manager.record_result(self.trial_number, trial_result)
        if self.trial is not None:
            try:
                self.trial.set_user_attr("score", float(self.composite_score))
                self.trial.set_user_attr("duration", float(self.elapsed_time))

                for key, value in self.kge_metrics.items():
                    try:
                        if key in ("score", "duration"):
                            continue
                        self.trial.set_user_attr(key, float(value))
                    except (ValueError, TypeError):
                        pass
            except Exception as exc:
                logger.debug(f"Failed to propagate trial attributes: {exc}")


def evaluate_trial_with_config(config: TrialEvaluationConfig) -> float:
    """Evaluate a trial using the DSLFM-only pipeline."""
    pipeline = TrialEvaluationPipeline(
        params=config.params,
        train_df=config.train_df,
        valid_df=config.valid_df,
        target_entity_ratio=config.target_entity_ratio,
        trial_number=config.trial_number,
        trial_output_root=config.trial_output_root,
        trial=config.trial,
        artifact_manager=config.artifact_manager,
    )
    score = pipeline.run()

    if config.trial is not None:
        metrics_payload = {}
        for key, value in pipeline.kge_metrics.items():
            try:
                metrics_payload[key] = float(value)
            except (ValueError, TypeError):
                pass

        metrics_payload.update(
            {
                "score": score,
                "trial_index": (
                    int(config.trial.number + 1)
                    if hasattr(config.trial, "number")
                    else config.trial_number
                ),
                "duration": float(pipeline.elapsed_time),
                "rank_block": float(pipeline.score_components.get("rank", 0.0)),
                "clf_block": float(
                    pipeline.score_components.get("classification", 0.0)
                ),
                "time_block": float(pipeline.score_components.get("efficiency", 0.0)),
            }
        )

        legacy_alias = {
            "score_composto": metrics_payload["score"],
            "kge_mrr": metrics_payload.get("mrr", 0.0),
            "kge_best_mrr": metrics_payload.get("best_mrr", 0.0),
            "kge_hits@1": metrics_payload.get("hits1", 0.0),
            "kge_hits@3": metrics_payload.get("hits3", 0.0),
            "kge_hits@10": metrics_payload.get("hits10", 0.0),
            "elapsed_time": metrics_payload["duration"],
        }
        try:
            for key, value in {**metrics_payload, **legacy_alias}.items():
                config.trial.set_user_attr(key, value)
        except Exception as attr_exc:
            logger.error(
                "Failed to attach metrics to trial attrs",
                error=str(attr_exc),
                component="TrialEvaluationPipeline",
            )

    return score


def evaluate_trial(
    params: dict[str, Any],
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    *,
    target_entity_ratio: float,
    trial_number: int,
    trial_output_root: Path,
    trial: Any | None = None,
    artifact_manager: TrialArtifactManager | None = None,
) -> float:
    """Legacy wrapper preserved for compatibility."""
    cfg = TrialEvaluationConfig(
        params=params,
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=target_entity_ratio,
        trial_number=trial_number,
        trial_output_root=trial_output_root,
        trial=trial,
        artifact_manager=artifact_manager,
    )
    return evaluate_trial_with_config(cfg)

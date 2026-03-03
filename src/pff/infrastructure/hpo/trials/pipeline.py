"""
DSLFM-KGC Trial Evaluation Pipeline.

Trains DSLFM-KGC (BERT + VAE + IBP + Probabilistic Circuits) and computes
multi-metric composite scores across ranking, classification and efficiency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
import polars as pl

from pff.domain.hpo.models import KGE_MODEL_DSLFM
from pff.domain.hpo.scoring import (
    build_weights_from_settings,
    compute_score,
    rename_metric_keys,
)
from pff.infrastructure.hpo.config_loader import (
    load_optimization_config,
    load_scoring_settings,
)
from pff.shared import logger
from pff.shared.acceleration.concurrency import ConcurrencyManager
from pff.shared.core.file_manager import FileManager
from pff.shared.determinism import set_global_seed
from pff.shared.ops.global_interrupt_manager import check_interruption
from pff.shared.system.cuda import is_cuda_available
from pff.shared.system.resource_manager import get_memory_safe_workers
from pff_rust import stable_hash

from .artifacts import TrialArtifactManager
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
        """Execute init.



        Args:

            params: Input value used by this callable.

            train_df: Input value used by this callable.

            valid_df: Input value used by this callable.

            target_entity_ratio: Input value used by this callable.

            trial_number: Input value used by this callable.

            trial_output_root: Input value used by this callable.

            trial: Optional input value.

            artifact_manager: Optional input value.

            enable_cross_validation: Optional input value.

            cv_fold_id: Optional input value.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.params = params
        self.train_df = train_df
        self.valid_df = valid_df
        self.target_entity_ratio = target_entity_ratio
        self.trial_number = trial_number
        self.trial_output_root = trial_output_root
        self.trial = trial
        self.artifact_manager = artifact_manager or TrialArtifactManager(base_dir=None)
        self.file_manager = FileManager()
        self.enable_cross_validation = enable_cross_validation
        self.cv_fold_id = cv_fold_id
        self.cv_settings = self._load_cv_settings()
        self.relation_id_policy = self._resolve_relation_id_policy()

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
            raw_study = getattr(getattr(self.trial, "study", None), "study_name", None)
            if raw_study is None:
                raw_study = getattr(self.artifact_manager, "study_name", None)
            study_name = (
                raw_study.strip()
                if isinstance(raw_study, str) and raw_study.strip()
                else None
            )

            status_path = (
                settings.OUTPUTS_DIR / "optimization" / "plots" / "live_status.json"
            )
            trial_status_path = (
                settings.OUTPUTS_DIR
                / "optimization"
                / "plots"
                / "live_status"
                / f"trial_{int(self.trial_number):06d}.json"
            )

            status: dict[str, Any] = {
                "trial_number": self.trial_number,
                "study_name": study_name,
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
            trial_status_path.parent.mkdir(parents=True, exist_ok=True)

            FileManager().save(status, trial_status_path)
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
        """Execute load cv settings.



        Returns:

            Return value produced by the callable.

        """

        defaults = {
            "cv_folds": 1,
            "cv_parallel": False,
            "cv_max_workers": 2,
            "cv_disable_when_cuda": True,
            "cv_disable_when_dataloader_workers": True,
            "cv_disable_when_auto_workers": True,
        }
        cfg = load_optimization_config("config/hpo/optimization.yaml", FileManager())
        defaults_cfg = cfg.get("defaults", {})
        parallel_cfg = cfg.get("parallel", {})
        cv_parallel_cfg: dict[str, Any] = {}
        if isinstance(parallel_cfg, dict):
            raw_cv_cfg = parallel_cfg.get("cv", {})
            if isinstance(raw_cv_cfg, dict):
                cv_parallel_cfg = raw_cv_cfg
        return {
            "cv_folds": int(defaults_cfg.get("cv_folds", defaults["cv_folds"])),
            "cv_parallel": bool(
                defaults_cfg.get("cv_parallel", defaults["cv_parallel"])
            ),
            "cv_max_workers": int(
                defaults_cfg.get("cv_max_workers", defaults["cv_max_workers"])
            ),
            "cv_disable_when_cuda": bool(
                cv_parallel_cfg.get(
                    "disable_when_cuda", defaults["cv_disable_when_cuda"]
                )
            ),
            "cv_disable_when_dataloader_workers": bool(
                cv_parallel_cfg.get(
                    "disable_when_dataloader_workers",
                    defaults["cv_disable_when_dataloader_workers"],
                )
            ),
            "cv_disable_when_auto_workers": bool(
                cv_parallel_cfg.get(
                    "disable_when_auto_workers",
                    defaults["cv_disable_when_auto_workers"],
                )
            ),
        }

    def _resolve_relation_id_policy(self) -> str:
        """Resolve relation-ID handling policy from params/config."""
        allowed = {"auto", "preserve_sparse", "remap_dense"}
        raw_policy = self.params.get("relation_id_policy")
        if raw_policy is None:
            cfg = load_optimization_config(
                "config/hpo/optimization.yaml", FileManager()
            )
            defaults_cfg = cfg.get("defaults", {}) if isinstance(cfg, dict) else {}
            raw_policy = defaults_cfg.get("relation_id_policy", "preserve_sparse")

        policy = str(raw_policy).strip().lower()
        if policy not in allowed:
            logger.warning(
                f"Invalid relation_id_policy={raw_policy!r}; falling back to 'preserve_sparse'."
            )
            return "preserve_sparse"
        return policy

    def _run_cross_validation(self) -> float:
        """Execute run cross validation.



        Returns:

            Return value produced by the callable.

        """

        cv_folds = self.cv_settings["cv_folds"]
        cv_parallel = self._resolve_cv_parallel(self.cv_settings["cv_parallel"])
        cv_workers = self.cv_settings["cv_max_workers"]
        logger.info(
            f"Iniciando cross-validation: folds={cv_folds} paralelo={cv_parallel}"
        )

        fold_indices = self._build_fold_indices(cv_folds)
        indices = np.arange(len(self.train_df))

        def _run_fold(
            fold_id: int,
            val_idx: np.ndarray,
        ) -> tuple[float, dict[str, float], float]:
            """Execute run fold.



            Args:

                fold_id: Input value used by this callable.

                val_idx: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

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
                check_interruption()
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
        logger.info(f"Validação cruzada concluída: score médio={mean_score:.4f}")
        return mean_score

    def _build_fold_indices(self, cv_folds: int) -> list[np.ndarray]:
        """Build deterministic, relation-stratified fold indices for CV."""
        rng = np.random.default_rng(self.trial_seed)
        all_indices = np.arange(len(self.train_df), dtype=np.int64)
        if cv_folds <= 1 or all_indices.size == 0:
            return [all_indices]

        relation_col = "p" if "p" in self.train_df.columns else None
        if relation_col is None:
            rng.shuffle(all_indices)
            return [
                fold.astype(np.int64, copy=False)
                for fold in np.array_split(all_indices, cv_folds)
            ]

        relation_values = self.train_df[relation_col].to_numpy()
        fold_chunks: list[list[np.ndarray]] = [[] for _ in range(cv_folds)]

        unique_relations = np.unique(relation_values)
        for relation in unique_relations:
            relation_idx = all_indices[relation_values == relation]
            if relation_idx.size == 0:
                continue
            rng.shuffle(relation_idx)
            relation_splits = np.array_split(relation_idx, cv_folds)
            for fold_id, split in enumerate(relation_splits):
                if split.size > 0:
                    fold_chunks[fold_id].append(split.astype(np.int64, copy=False))

        fold_indices: list[np.ndarray] = []
        for chunks in fold_chunks:
            if chunks:
                merged = np.concatenate(chunks)
                rng.shuffle(merged)
            else:
                merged = np.empty(0, dtype=np.int64)
            fold_indices.append(merged)
        return fold_indices

    def _resolve_cv_parallel(self, requested_parallel: bool) -> bool:
        """Execute resolve cv parallel.



        Args:

            requested_parallel: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not requested_parallel:
            return False

        if self.cv_settings["cv_disable_when_cuda"] and is_cuda_available():
            logger.debug(
                "Parallel cross-validation disabled: CUDA detected; "
                "single-GPU folds should run sequentially."
            )
            return False

        try:
            requested_workers = int(self.params.get("num_workers", 0))
        except (TypeError, ValueError):
            requested_workers = 0

        if (
            self.cv_settings["cv_disable_when_dataloader_workers"]
            and requested_workers > 0
        ):
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
        if self.cv_settings["cv_disable_when_auto_workers"] and auto_workers > 0:
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
        check_interruption()
        start = time.time()
        if self.kge_model_dir is None:
            raise RuntimeError("Model directory not initialized")
        kge_checkpoint_dir = self.kge_model_dir / "checkpoints"
        self.file_manager.ensure_dir(kge_checkpoint_dir)

        train_triples, valid_triples, num_entities, num_relations, relation_names = (
            self._prepare_kge_triples()
        )

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
                study_name_override=getattr(self.artifact_manager, "study_name", None),
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

        raw_metrics = self._merge_kge_metrics(kge_stats)
        self.kge_metrics = rename_metric_keys(raw_metrics)
        self.kge_checkpoint_path = checkpoint_path
        self.model_paths["dslfm"] = str(checkpoint_path)

    def _prepare_kge_triples(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int, int, list[str]]:
        """Execute prepare kge triples.



        Returns:

            Return value produced by the callable.

        """

        sparse_ids_ready = self._prepare_kge_triples_with_sparse_relations()
        if sparse_ids_ready is not None:
            return sparse_ids_ready

        entity_ids = pl.concat(
            [
                self.train_df["s"],
                self.train_df["o"],
                self.valid_df["s"],
                self.valid_df["o"],
            ]
        )
        relation_ids = pl.concat([self.train_df["p"], self.valid_df["p"]])
        contiguous = self._resolve_contiguous_space(entity_ids, relation_ids)
        if contiguous is not None:
            num_entities, num_relations = contiguous
            train_triples = np.asarray(
                self.train_df.select(["s", "p", "o"]).to_numpy(),
                dtype=np.int64,
            )
            valid_triples = np.asarray(
                self.valid_df.select(["s", "p", "o"]).to_numpy(),
                dtype=np.int64,
            )
            relation_names = self._resolve_relation_names(
                num_relations=num_relations,
                fallback=[str(i) for i in range(num_relations)],
            )
            return (
                train_triples,
                valid_triples,
                num_entities,
                num_relations,
                relation_names,
            )
        return self._prepare_mapped_kge_triples()

    def _resolve_relation_names(
        self, *, num_relations: int, fallback: list[str]
    ) -> list[str]:
        """Resolve relation names with semantic labels when preprocessing maps exist."""
        relation_names = self._load_relation_names_from_preprocessing_map(
            num_relations=num_relations
        )
        if relation_names is not None:
            return relation_names
        return fallback

    def _load_relation_names_from_preprocessing_map(
        self, *, num_relations: int
    ) -> list[str] | None:
        """Load relation names from preprocessing map to preserve semantic labels."""
        candidates: list[Path] = []

        override = self.params.get("relation_map_path")
        if isinstance(override, str) and override.strip():
            candidates.append(Path(override.strip()))

        try:
            from pff.domain.kg.preprocessing.config import PreprocessingConfig

            pre_cfg = PreprocessingConfig.from_yaml()
            candidates.append(
                Path(pre_cfg.output_dir)
                / f"relation_map_{stable_hash('splits')}.parquet"
            )
        except Exception as exc:
            logger.debug(
                f"Unable to resolve preprocessing output_dir for relation map: {exc}"
            )

        candidates.append(
            Path("outputs/preprocessing")
            / f"relation_map_{stable_hash('splits')}.parquet"
        )

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if not self.file_manager.exists(resolved):
                continue
            try:
                payload = self.file_manager.read(resolved)
            except Exception as exc:
                logger.debug(f"Failed reading relation map candidate {resolved}: {exc}")
                continue
            if hasattr(payload, "lazyframe"):
                relation_map = payload.lazyframe().collect()
            elif isinstance(payload, pl.LazyFrame):
                relation_map = payload.collect()
            elif isinstance(payload, pl.DataFrame):
                relation_map = payload
            else:
                logger.debug(f"Unexpected relation map payload type: {type(payload)}")
                continue

            id_col = "relation_id" if "relation_id" in relation_map.columns else "id"
            label_col = "relation" if "relation" in relation_map.columns else "label"
            if (
                id_col not in relation_map.columns
                or label_col not in relation_map.columns
            ):
                logger.debug(
                    f"Relation map missing expected columns at {resolved}: {relation_map.columns}"
                )
                continue

            normalized = relation_map.select(
                [
                    pl.col(id_col).cast(pl.Int64).alias("id"),
                    pl.col(label_col).cast(pl.Utf8).alias("label"),
                ]
            ).sort("id")

            if num_relations <= 0:
                return []
            if normalized.height == 0:
                continue

            bounded = normalized.filter(
                (pl.col("id") >= 0) & (pl.col("id") < int(num_relations))
            )
            if bounded.height == 0:
                continue

            resolved_names = [str(i) for i in range(num_relations)]
            for rel_id, label in bounded.iter_rows():
                resolved_names[int(rel_id)] = str(label)
            return resolved_names

        return None

    @staticmethod
    def _is_contiguous_id_space(series: pl.Series) -> tuple[bool, int]:
        """Execute is contiguous id space.



        Args:

            series: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if series.null_count() != 0 or not series.dtype.is_integer():
            return False, 0
        min_value = series.min()
        max_value = series.max()
        if min_value != 0 or max_value is None:
            return False, 0
        unique_count = int(series.n_unique())  # type: ignore[arg-type]
        max_int = int(cast(int | float, max_value))
        if max_int != unique_count - 1:
            return False, 0
        return True, max_int + 1

    def _resolve_contiguous_space(
        self, entity_ids: pl.Series, relation_ids: pl.Series
    ) -> tuple[int, int] | None:
        """Execute resolve contiguous space.



        Args:

            entity_ids: Input value used by this callable.

            relation_ids: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        entity_ok, num_entities = self._is_contiguous_id_space(entity_ids)
        relation_ok, num_relations = self._is_contiguous_id_space(relation_ids)
        if entity_ok and relation_ok:
            return num_entities, num_relations
        return None

    @staticmethod
    def _is_non_negative_integer_series(series: pl.Series) -> tuple[bool, int]:
        """Check whether series contains non-negative integer IDs and return max+1."""
        if series.null_count() != 0 or not series.dtype.is_integer():
            return False, 0
        min_value = series.min()
        max_value = series.max()
        if min_value is None or max_value is None:
            return False, 0
        min_int = int(cast(int | float, min_value))
        if min_int < 0:
            return False, 0
        return True, int(cast(int | float, max_value)) + 1

    def _prepare_kge_triples_with_sparse_relations(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int, int, list[str]] | None:
        """Prepare triples while preserving sparse integer relation IDs when configured."""
        if self.relation_id_policy == "remap_dense":
            return None

        required_cols = ("s", "p", "o")
        if not all(col in self.train_df.columns for col in required_cols):
            return None
        if not all(col in self.valid_df.columns for col in required_cols):
            return None

        entity_ids = pl.concat(
            [
                self.train_df["s"],
                self.train_df["o"],
                self.valid_df["s"],
                self.valid_df["o"],
            ]
        )
        relation_ids = pl.concat([self.train_df["p"], self.valid_df["p"]])

        entities_ok, entity_space = self._is_non_negative_integer_series(entity_ids)
        relations_ok, relation_space = self._is_non_negative_integer_series(
            relation_ids
        )
        if not entities_ok or not relations_ok:
            return None

        relation_contiguous, relation_contiguous_space = self._is_contiguous_id_space(
            relation_ids
        )
        if relation_contiguous:
            resolved_relation_space = relation_contiguous_space
        elif self.relation_id_policy == "auto":
            unique_relations = int(relation_ids.n_unique())  # type: ignore[arg-type]
            density = unique_relations / max(1, relation_space)
            if density < 0.35:
                return None
            resolved_relation_space = relation_space
        else:
            unique_relations = int(relation_ids.n_unique())  # type: ignore[arg-type]
            logger.info(
                "Preservando IDs esparsos de relacao para evitar degradacao de MRR: "
                f"relacoes_unicas={unique_relations}, espaco_ids={relation_space}"
            )
            resolved_relation_space = relation_space

        entity_contiguous, entity_contiguous_space = self._is_contiguous_id_space(
            entity_ids
        )
        if entity_contiguous:
            train_triples = np.asarray(
                self.train_df.select(["s", "p", "o"]).to_numpy(),
                dtype=np.int64,
            )
            valid_triples = np.asarray(
                self.valid_df.select(["s", "p", "o"]).to_numpy(),
                dtype=np.int64,
            )
            resolved_entity_space = entity_contiguous_space
        else:
            train_triples, valid_triples, resolved_entity_space = (
                self._prepare_entity_mapped_relation_preserved_kge_triples()
            )
            resolved_entity_space = min(resolved_entity_space, entity_space)

        relation_names = self._resolve_relation_names(
            num_relations=int(resolved_relation_space),
            fallback=[str(i) for i in range(int(resolved_relation_space))],
        )
        return (
            train_triples,
            valid_triples,
            int(resolved_entity_space),
            int(resolved_relation_space),
            relation_names,
        )

    def _prepare_entity_mapped_relation_preserved_kge_triples(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Map only entity IDs to dense space while keeping relation IDs untouched."""
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
        entity_map = pl.DataFrame({"label": entity_labels}).with_row_index("id")
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
                entity_map,
                left_on="o",
                right_on="label",
                how="left",
                maintain_order="left",
            )
            .rename({"id": "o_id"})
            .select(["__split", "s_id", "p", "o_id"])
        )
        train_triples = np.asarray(
            mapped.filter(pl.col("__split") == "train")
            .select(["s_id", "p", "o_id"])
            .to_numpy(),
            dtype=np.int64,
        )
        valid_triples = np.asarray(
            mapped.filter(pl.col("__split") == "valid")
            .select(["s_id", "p", "o_id"])
            .to_numpy(),
            dtype=np.int64,
        )
        return train_triples, valid_triples, int(entity_map.height)

    def _prepare_mapped_kge_triples(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int, int, list[str]]:
        """Execute prepare mapped kge triples.



        Returns:

            Return value produced by the callable.

        """

        import numpy as np

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
        resolved_relation_names = self._resolve_relation_names(
            num_relations=int(relation_map.height),
            fallback=[str(value) for value in relation_names],
        )
        return (
            train_triples,
            valid_triples,
            int(entity_map.height),
            int(relation_map.height),
            resolved_relation_names,
        )

    @staticmethod
    def _merge_kge_metrics(kge_stats: dict[str, Any]) -> dict[str, Any]:
        """Execute merge kge metrics.



        Args:

            kge_stats: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        best_metrics = kge_stats.get("best_metrics", {})
        raw_metrics = kge_stats.get("final_metrics", {})
        if best_metrics:
            for key, value in best_metrics.items():
                if key not in raw_metrics or raw_metrics[key] == 0:
                    raw_metrics[key] = value
        best_mrr = kge_stats.get("best_val_mrr", raw_metrics.get("mrr", 0.0))
        raw_metrics["best_mrr"] = best_mrr
        if best_mrr > raw_metrics.get("mrr", 0.0):
            raw_metrics["mrr"] = best_mrr
        best_mcc = kge_stats.get("best_val_mcc", raw_metrics.get("mcc", 0.0))
        raw_metrics["best_mcc"] = best_mcc
        if best_mcc > raw_metrics.get("mcc", 0.0):
            raw_metrics["mcc"] = best_mcc
        return raw_metrics

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

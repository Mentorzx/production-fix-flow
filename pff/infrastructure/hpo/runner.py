"""
Core Hyperparameter Optimization (DSLFM/PC-only).

This module provides a single entry point `optimize_kg_hyperparameters`
that trains and scores only the DSLFM model (lógica + Probabilistic Circuits).
Legacy ensemble paths were removed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from pff import settings
from pff.config import OPTIMIZATION_CONFIG_PATH, KG_PIPELINE_CONFIG_PATH
from pff.shared import logger
from pff.shared.acceleration.asyncio_runner import run_coroutine_sync
from pff.shared.core.file_manager import FileManager, ParquetBundle
from pff.infrastructure.persistence.db.connection import close_connection_pool
from pff.application.ports.hpo import HpoRunnerPort

from .trials.artifacts import TrialArtifactManager
from .trials.data_loader import load_preprocessed_from_postgres, load_synthetic_kg_data
from pff.domain.hpo.selection import select_best_trials
from pff.domain.hpo.models import KGE_MODEL_DSLFM, resolve_kge_model
from .trials.archive import archive_and_reset_trials
from .trials.objective import kg_objective
from .trials.study import create_study_and_run
from pff.domain.hpo.search_space import TuningConfigBuilder
from pff.infrastructure.hpo.config_loader import load_hpo_defaults, load_scoring_weights
from pff.infrastructure.hpo.config_updater import (
    DataScaleProfile,
    export_hpo_summary,
    update_dslfm_config,
)
from pff.infrastructure.hpo.trials.postgres_store import HpoPostgresStore


DEFAULT_KGE_MODEL = KGE_MODEL_DSLFM
_checkpoint_file_manager = FileManager()


class HpoRunner(HpoRunnerPort):
    """Infrastructure runner that executes the HPO pipeline."""

    def run(
        self,
        *,
        n_trials: int,
        strategy: str,
        enable_mlflow: bool,
        enable_visualization: bool,
        study_name: str | None,
        output_dir: Path | None,
        target_entity_ratio: float,
        kge_model: str,
        use_synthetic_if_dslfm: bool,
        no_update_config: bool,
        no_bert: bool,
        resume_mode: bool | None,
        reset_state: bool,
    ) -> dict[str, Any]:
        return optimize_kg_hyperparameters(
            n_trials=n_trials,
            strategy=strategy,
            enable_mlflow=enable_mlflow,
            enable_visualization=enable_visualization,
            study_name=study_name,
            output_dir=output_dir,
            target_entity_ratio=target_entity_ratio,
            kge_model=kge_model,
            use_synthetic_if_dslfm=use_synthetic_if_dslfm,
            no_update_config=no_update_config,
            no_bert=no_bert,
            resume_mode=resume_mode,
            reset_state=reset_state,
        )


@dataclass
class HPOMemoryConfig:
    """Configuration for persistent HPO memory."""

    enabled: bool = True
    top_k_trials: int = 5
    warmstart_trials: int = 3
    storage_subdir: str = "hpo_replay"
    min_score_delta: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> HPOMemoryConfig:
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            top_k_trials=int(data.get("top_k_trials", 5)),
            warmstart_trials=int(data.get("warmstart_trials", 3)),
            storage_subdir=str(data.get("storage_subdir", "hpo_replay")),
            min_score_delta=float(data.get("min_score_delta", 0.0)),
        )


def _load_hpo_memory_config(file_manager: FileManager | None = None) -> HPOMemoryConfig:
    """Load HPO memory configuration from config/hpo/optimization.yaml."""
    fm = file_manager or FileManager()
    config_path = OPTIMIZATION_CONFIG_PATH
    try:
        payload = fm.read(config_path)
        raw_config = (
            payload.to_native() if isinstance(payload, ParquetBundle) else payload or {}
        )
        memory_config = (
            raw_config.get("hpo_memory", {}) if isinstance(raw_config, dict) else {}
        )
        return HPOMemoryConfig.from_dict(memory_config)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to load HPO optimization config: {exc}")
        return HPOMemoryConfig()


class _TrialSerializationMixin:
    """Shared Optuna distribution serialization helpers (Template Method)."""

    @staticmethod
    def _serialize_distributions(distributions: dict[str, Any]) -> dict[str, Any]:
        """Serialize Optuna distributions to JSON-friendly format."""
        try:
            import optuna
            from optuna.distributions import CategoricalDistribution
        except Exception:  # pragma: no cover
            return {}
        serialized = {}
        for name, dist in distributions.items():
            if isinstance(dist, optuna.distributions.FloatDistribution):
                serialized[name] = {
                    "type": "float",
                    "low": dist.low,
                    "high": dist.high,
                    "log": dist.log,
                    "step": dist.step,
                }
            elif isinstance(dist, optuna.distributions.IntDistribution):
                serialized[name] = {
                    "type": "int",
                    "low": dist.low,
                    "high": dist.high,
                    "log": dist.log,
                    "step": dist.step,
                }
            elif isinstance(dist, CategoricalDistribution):
                serialized[name] = {
                    "type": "categorical",
                    "choices": dist.choices,
                }
        return serialized

    @staticmethod
    def _deserialize_distributions(payload: dict[str, Any]) -> dict[str, Any]:
        """Deserialize distributions from stored JSON."""
        try:
            from optuna.distributions import (
                CategoricalDistribution,
                FloatDistribution,
                IntDistribution,
            )
        except Exception:  # pragma: no cover
            return {}
        distributions = {}
        for name, dist_payload in payload.items():
            if not isinstance(dist_payload, dict):
                continue
            dist_type = dist_payload.get("type")
            if dist_type == "float":
                distributions[name] = FloatDistribution(
                    low=float(dist_payload.get("low")),
                    high=float(dist_payload.get("high")),
                    log=bool(dist_payload.get("log", False)),
                    step=dist_payload.get("step"),
                )
            elif dist_type == "int":
                distributions[name] = IntDistribution(
                    low=int(dist_payload.get("low")),
                    high=int(dist_payload.get("high")),
                    log=bool(dist_payload.get("log", False)),
                    step=dist_payload.get("step"),
                )
            elif dist_type == "categorical":
                distributions[name] = CategoricalDistribution(
                    choices=list(dist_payload.get("choices", []))
                )
        return distributions

    @staticmethod
    def _params_match(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
        if set(lhs.keys()) != set(rhs.keys()):
            return False
        for key, left_val in lhs.items():
            right_val = rhs.get(key)
            if isinstance(left_val, float) and isinstance(right_val, float):
                if abs(left_val - right_val) > 1e-9:
                    return False
            elif left_val != right_val:
                return False
        return True


class PersistentBestTrialMemory(_TrialSerializationMixin):
    """Persist best trial metrics to warm-start future HPO runs."""

    def __init__(
        self,
        output_dir: Path,
        config: HPOMemoryConfig,
        *,
        study_name: str | None = None,
        store: HpoPostgresStore | None = None,
        file_manager: FileManager | None = None,
    ):
        self.config = config
        self.file_manager = file_manager or FileManager()
        self.study_name = study_name
        self.store = store
        if self.store is None:
            raise ValueError("HPO memory persistence requires a Postgres store")
        self.entries: list[dict[str, Any]] = self._load_entries()

    def record_trial(
        self, study, trial, trial_result: dict[str, Any] | None = None
    ) -> None:
        """Record a completed trial with metrics into the persistent memory."""
        if not self.config.enabled:
            return
        try:
            from optuna.trial import TrialState
        except Exception:  # pragma: no cover
            return
        if getattr(trial, "state", None) != TrialState.COMPLETE:
            return
        if trial.value is None:
            return
        if self.entries and len(self.entries) >= self.config.top_k_trials:
            best_value = float(self.entries[0]["value"])
            if float(trial.value) + self.config.min_score_delta < best_value and all(
                entry["value"] >= float(trial.value) for entry in self.entries
            ):
                return
        metrics: dict[str, float] = {}
        model_metrics: dict[str, float] = {}
        if trial_result:
            raw_metrics = trial_result.get("metrics", {})
            raw_model_metrics = trial_result.get("model_metrics", {})
            if isinstance(raw_metrics, dict):
                metrics = self._coerce_metrics(raw_metrics)
            if isinstance(raw_model_metrics, dict):
                model_metrics = self._coerce_metrics(raw_model_metrics)
        entry = {
            "study_name": getattr(study, "study_name", ""),
            "trial_number": trial.number,
            "value": float(trial.value),
            "params": dict(trial.params),
            "distributions": self._serialize_distributions(
                getattr(trial, "distributions", {}) or {}
            ),
            "metrics": metrics,
            "model_metrics": model_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.entries.append(entry)
        self.entries = sorted(
            self.entries, key=lambda item: item["value"], reverse=True
        )[: self.config.top_k_trials]
        self._persist()

    def warmstart_study(self, study) -> int:
        """Inject best trials as completed seeds into a new Optuna study."""
        if not self.config.enabled or not self.entries:
            return 0
        try:
            import optuna
            from optuna.trial import TrialState
        except Exception:  # pragma: no cover
            return 0
        added = 0
        existing_trials = [
            trial
            for trial in getattr(study, "trials", [])
            if getattr(trial, "state", None)
        ]
        for entry in self.entries[: self.config.warmstart_trials]:
            if any(
                self._params_match(trial.params, entry["params"])
                for trial in existing_trials
            ):
                continue
            distributions = self._deserialize_distributions(
                entry.get("distributions", {})
            )
            try:
                frozen = optuna.create_trial(
                    state=TrialState.COMPLETE,
                    value=float(entry["value"]),
                    params=entry["params"],
                    distributions=distributions,
                    system_attrs={"warmstart_seed": True},
                )
                study.add_trial(frozen)
                added += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to warm-start trial replay: {exc}")
        if added > 0:
            logger.debug(f"warmstart_trials_loaded n={added}")
        return added

    def _load_entries(self) -> list[dict[str, Any]]:
        if self.store is None or not self.study_name:
            raise ValueError(
                "HPO memory persistence requires a Postgres store and study name"
            )
        try:
            payload = run_coroutine_sync(
                self.store.load_memory_entries(self.study_name)
            )
            return list(payload or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to load HPO memory from Postgres: {exc}")
            return []

    def _persist(self) -> None:
        if self.store is None or not self.study_name:
            raise ValueError(
                "HPO memory persistence requires a Postgres store and study name"
            )
        try:
            run_coroutine_sync(
                self.store.upsert_memory_entries(self.study_name, self.entries)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to persist HPO memory to Postgres: {exc}")

    def _coerce_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


class BestModelSaverCallback(_TrialSerializationMixin):
    """Optuna callback to persist the best trial parameters and cleanup artifacts."""

    def __init__(
        self,
        output_dir: Path,
        memory: PersistentBestTrialMemory | None = None,
        artifact_manager=None,
        trial_runs_dir: Path | None = None,
        study_name: str | None = None,
        store: HpoPostgresStore | None = None,
        file_manager: FileManager | None = None,
    ) -> None:
        self.file_manager = file_manager or FileManager()
        self.output_dir = Path(output_dir)
        self.file_manager.ensure_dir(self.output_dir)
        self.memory = memory
        self.artifact_manager = artifact_manager
        self.trial_runs_dir = trial_runs_dir
        self.study_name = study_name
        self.store = store
        self.best_value: float = float("-inf")
        self.best_params: dict[str, Any] = {}

    def __call__(self, study, trial) -> None:
        value = getattr(trial, "value", None)
        if value is None:
            return
        try:
            numeric_value = float(value)
        except Exception:
            return

        if self.memory is not None:
            try:
                user_attrs = dict(getattr(trial, "user_attrs", {}) or {})
                numeric_user_attrs = {
                    key: val
                    for key, val in user_attrs.items()
                    if isinstance(val, (int, float))
                }
                self.memory.record_trial(
                    study,
                    trial,
                    {
                        "metrics": numeric_user_attrs,
                        "model_metrics": numeric_user_attrs,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to record trial in memory: {exc}")

        if numeric_value <= self.best_value:
            self._cleanup_trial_dir(trial)
            return

        self.best_value = numeric_value
        self.best_params = dict(getattr(trial, "params", {}))
        self._persist_best_params()

    def _persist_best_params(self) -> None:
        try:
            if self.store is not None and self.study_name:
                run_coroutine_sync(
                    self.store.upsert_best_params(
                        self.study_name,
                        self.best_params,
                        self.best_value,
                    )
                )
                logger.info("Parametros otimizados salvos no Postgres")
                return
            raise ValueError("HPO best params persistence requires a Postgres store")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to persist best_params: {exc}")

    def _cleanup_trial_dir(self, trial) -> None:
        trial_number = int(getattr(trial, "number", -1))
        if trial_number < 0:
            return
        base_dir = self.trial_runs_dir
        if base_dir is None and self.artifact_manager is not None:
            base_dir = getattr(self.artifact_manager, "base_dir", None)
        if base_dir is None:
            return
        trial_dir = Path(base_dir) / f"trial_{trial_number:04d}"
        self.file_manager.delete_directory(trial_dir, ignore_errors=True)

    def _coerce_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

    @staticmethod
    def _deserialize_distributions(payload: dict[str, Any]) -> dict[str, Any]:
        """Deserialize distributions from stored JSON."""
        try:
            from optuna.distributions import (
                CategoricalDistribution,
                FloatDistribution,
                IntDistribution,
            )
        except Exception:  # pragma: no cover
            return {}
        distributions = {}
        for name, dist_payload in payload.items():
            if not isinstance(dist_payload, dict):
                continue
            dist_type = dist_payload.get("type")
            if dist_type == "float":
                distributions[name] = FloatDistribution(
                    low=float(dist_payload.get("low")),
                    high=float(dist_payload.get("high")),
                    log=bool(dist_payload.get("log", False)),
                    step=dist_payload.get("step"),
                )
            elif dist_type == "int":
                distributions[name] = IntDistribution(
                    low=int(dist_payload.get("low")),
                    high=int(dist_payload.get("high")),
                    log=bool(dist_payload.get("log", False)),
                    step=dist_payload.get("step"),
                )
            elif dist_type == "categorical":
                distributions[name] = CategoricalDistribution(
                    choices=list(dist_payload.get("choices", []))
                )
        return distributions

    @staticmethod
    def _params_match(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
        if set(lhs.keys()) != set(rhs.keys()):
            return False
        for key, left_val in lhs.items():
            right_val = rhs.get(key)
            if isinstance(left_val, float) and isinstance(right_val, float):
                if abs(left_val - right_val) > 1e-9:
                    return False
            elif left_val != right_val:
                return False
        return True


def optimize_kg_hyperparameters(
    n_trials: int = 100,
    strategy: str = "optuna",
    enable_mlflow: bool = False,
    enable_visualization: bool = False,
    study_name: str | None = None,
    output_dir: Path | None = None,
    target_entity_ratio: float = 0.7,
    kge_model: str = DEFAULT_KGE_MODEL,
    use_synthetic_if_dslfm: bool = False,
    no_update_config: bool = False,
    no_bert: bool = False,
    resume_mode: bool | None = None,
    reset_state: bool = False,
) -> dict[str, Any]:
    """Optimize DSLFM hyperparameters using real KG data (no ensemble)."""
    kge_model = resolve_kge_model(kge_model)
    if use_synthetic_if_dslfm:
        logger.warning("Synthetic DSLFM enabled; using synthetic data for trial")
    if no_bert:
        logger.info("BERT desabilitado para HPO via CLI")

    logger.info(
        f"hpo_dslfm_iniciado kge_model={kge_model.upper()} n_trials={n_trials} strategy={strategy}"
    )
    file_manager = FileManager()

    if use_synthetic_if_dslfm:
        train_df, valid_df, data_info = load_synthetic_kg_data(
            file_manager,
            config_path=OPTIMIZATION_CONFIG_PATH,
        )
    else:
        train_df, valid_df, data_info = load_preprocessed_from_postgres(
            file_manager,
            require_preprocessed=True,
            auto_populate_if_missing=True,
            config_path=KG_PIPELINE_CONFIG_PATH,
        )
    logger.info(
        f"dados_carregados train={data_info['n_train']:,} valid={data_info['n_valid']:,} "
        f"entidades={data_info['n_entities']:,} predicados={data_info['n_predicates']} "
        f"fonte={data_info.get('source', 'unknown')}"
    )

    tuning_defaults = load_hpo_defaults(file_manager)
    if no_bert:
        tuning_defaults = {**tuning_defaults, "use_bert": False}
    tuning_config = TuningConfigBuilder(tuning_defaults).build()
    hpo_ranges = {
        "kge": {
            "embedding_dim": {"choices": list(tuning_config.embedding_dim_choices)},
            "batch_size": {
                "low": tuning_config.batch_size_low,
                "high": tuning_config.batch_size_high,
            },
            "negative_sample_size": {
                "low": tuning_config.negative_sample_size_low,
                "high": tuning_config.negative_sample_size_high,
            },
            "adversarial_temperature": {
                "low": tuning_config.adversarial_temperature_low,
                "high": tuning_config.adversarial_temperature_high,
            },
            "learning_rate": {
                "low": tuning_config.learning_rate_low,
                "high": tuning_config.learning_rate_high,
            },
            "self_adversarial": {
                "choices": list(tuning_config.self_adversarial_choices)
            },
            "use_bert_default": bool(tuning_config.use_bert_default),
        },
        "logic": {
            "lambda_logic": {
                "low": tuning_config.lambda_logic_low,
                "high": tuning_config.lambda_logic_high,
            },
            "t_norm": {"choices": list(tuning_config.t_norm_choices)},
            "attr_hidden_dim": {"choices": list(tuning_config.attr_hidden_dim_choices)},
        },
        "pc": {
            "lambda_pc": {
                "low": tuning_config.lambda_pc_low,
                "high": tuning_config.lambda_pc_high,
            },
            "pruning_threshold": {
                "low": tuning_config.pruning_threshold_low,
                "high": tuning_config.pruning_threshold_high,
            },
            "rebuild_every": {
                "low": tuning_config.rebuild_every_low,
                "high": tuning_config.rebuild_every_high,
            },
            "max_circuit_depth": {
                "choices": list(tuning_config.max_circuit_depth_choices)
            },
        },
        "regularization": {"lambda_sum_cap": tuning_config.lambda_sum_cap},
        "contrastive": {
            "temperature_low": tuning_config.contrastive_temperature_low,
            "temperature_high": tuning_config.contrastive_temperature_high,
            "num_global_negatives_low": tuning_config.num_global_negatives_low,
            "num_global_negatives_high": tuning_config.num_global_negatives_high,
        },
        "architecture": {
            "kl_weight_low": tuning_config.kl_weight_low,
            "kl_weight_high": tuning_config.kl_weight_high,
        },
    }

    if output_dir:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = settings.OUTPUTS_DIR / output_dir
    else:
        output_dir = settings.OUTPUTS_DIR / "optimization" / "kg_dslfm"
    FileManager.ensure_dir(output_dir)

    checkpoint_store = HpoPostgresStore(file_manager=file_manager)
    checkpoint_key = f"hpo::{output_dir.resolve()}"
    checkpoint_data = _load_checkpoint(
        None,
        store=checkpoint_store,
        checkpoint_key=checkpoint_key,
    )
    if study_name is None and checkpoint_data:
        checkpoint_study_name = checkpoint_data.get("study_name")
        if isinstance(checkpoint_study_name, str) and checkpoint_study_name.strip():
            study_name = checkpoint_study_name.strip()
    study_name = study_name or f"pff_kg_optimization_{int(time.time())}"

    safe_study = study_name.replace(" ", "_").replace("/", "_")
    work_dir = settings.CACHE_DIR / "hpo" / safe_study
    FileManager.ensure_dir(work_dir)
    checkpoint_path = work_dir / "checkpoint.json"
    storage_path = work_dir / "optuna_study.db"
    artifact_manager = TrialArtifactManager(
        base_dir=None,
        study_name=study_name,
        store=checkpoint_store,
        file_manager=file_manager,
    )

    storage_exists = file_manager.exists(storage_path)
    checkpoint_exists = checkpoint_data is not None
    auto_resume = storage_exists or checkpoint_exists
    resolved_resume_mode = auto_resume if resume_mode is None else bool(resume_mode)

    if reset_state:
        archive_and_reset_trials(
            work_dir,
            study_name=study_name,
            top_n=5,
            store=checkpoint_store,
            file_manager=file_manager,
        )
        checkpoint_data = None
        resolved_resume_mode = False
        logger.info(f"hpo_reset_ativo=true output_dir={output_dir}")
    else:
        logger.debug(
            "hpo_resume_decisao "
            f"resume_mode={resolved_resume_mode} storage_exists={storage_exists} "
            f"checkpoint_exists={checkpoint_exists} output_dir={output_dir}"
        )

    trial_runs_dir = work_dir / "trials"

    def objective_fn(trial):
        return kg_objective(
            trial,
            train_df=train_df,
            valid_df=valid_df,
            target_entity_ratio=target_entity_ratio,
            trial_runs_dir=trial_runs_dir,
            hpo_ranges=hpo_ranges,
            file_manager=file_manager,
            artifact_manager=artifact_manager,
        )

    hpo_memory_config = _load_hpo_memory_config(file_manager)
    trial_memory = PersistentBestTrialMemory(
        work_dir,
        hpo_memory_config,
        study_name=study_name,
        store=checkpoint_store,
        file_manager=file_manager,
    )
    expected_trials = n_trials
    if checkpoint_data:
        expected_trials_candidate = checkpoint_data.get("expected_trials")
        try:
            expected_trials = max(expected_trials, int(expected_trials_candidate))
        except Exception:
            expected_trials = n_trials

    result = create_study_and_run(
        study_name=study_name,
        storage_path=storage_path,
        checkpoint_path=checkpoint_path,
        checkpoint_key=checkpoint_key,
        checkpoint_store=checkpoint_store,
        output_dir=output_dir,
        work_dir=work_dir,
        n_trials=n_trials,
        expected_trials=expected_trials,
        resume_mode=resolved_resume_mode,
        checkpoint_data=checkpoint_data,
        hpo_memory_config=hpo_memory_config,
        trial_memory=trial_memory,
        warmstart_callback=trial_memory.warmstart_study,
        objective_fn=objective_fn,
        artifact_manager=artifact_manager,
        enable_mlflow=enable_mlflow,
        file_manager=file_manager,
    )

    scoring_weights = load_scoring_weights(file_manager)
    selection = select_best_trials(result.get("study"), weights=scoring_weights)
    result["multi_objective"] = selection
    result["optuna_best_value"] = result.get("best_value")
    result["optuna_best_params"] = dict(result.get("best_params") or {})
    best_tradeoff = selection.get("best_tradeoff")
    if best_tradeoff:
        result["best_params"] = best_tradeoff.get("params", {})
        result["best_value"] = best_tradeoff.get("score_time")
        result["best_value_tradeoff"] = best_tradeoff.get("tradeoff_score")
        summary_path = output_dir / "multi_objective_summary.json"
        try:
            file_manager.save(selection, summary_path)
            result["multi_objective_summary"] = summary_path
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to persist multi-objective summary: {exc}")
    best_time = selection.get("best_time_aware") or {}
    best_quality = selection.get("best_quality") or {}
    if best_time:
        result["best_value_time_aware"] = best_time.get("score_time")
    if best_quality:
        result["best_value_quality"] = best_quality.get("score_quality")

    result["real_data_info"] = data_info
    result["kge_model"] = kge_model

    try:
        summary_path = export_hpo_summary(result, output_dir, file_manager=file_manager)
        result["hpo_summary_path"] = summary_path
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Falha ao exportar sumario HPO: {exc}")

    if no_update_config:
        logger.info("Atualizacao automatica do config DSLFM desabilitada")
    else:
        best_params = result.get("best_params") or {}
        if best_params:
            try:
                data_profile = DataScaleProfile.from_data_info(data_info)
                update_result = update_dslfm_config(
                    best_params=best_params,
                    data_profile=data_profile,
                    file_manager=file_manager,
                )
                result["config_update"] = update_result
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Falha ao atualizar config DSLFM: {exc}")
        else:
            logger.warning("No best parameters found; DSLFM config not updated")

    try:
        run_coroutine_sync(close_connection_pool())
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to close database connection pool: {exc}")

    return result


def _load_checkpoint(
    checkpoint_path: Path | None,
    *,
    store: HpoPostgresStore | None = None,
    checkpoint_key: str | None = None,
) -> dict[str, Any] | None:
    """Load checkpoint using Postgres store when available."""
    if store is None or checkpoint_key is None:
        raise ValueError("HPO checkpoints require a Postgres store and checkpoint key")
    try:
        return run_coroutine_sync(store.load_checkpoint(checkpoint_key))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to load checkpoint from Postgres: {exc}")
        return None


def _write_checkpoint(
    checkpoint_path: Path | None,
    payload: dict[str, Any],
    *,
    store: HpoPostgresStore | None = None,
    checkpoint_key: str | None = None,
) -> None:
    """Write checkpoint to Postgres when available."""
    if store is None or checkpoint_key is None:
        raise ValueError("HPO checkpoints require a Postgres store and checkpoint key")
    try:
        run_coroutine_sync(store.upsert_checkpoint(checkpoint_key, payload))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to persist checkpoint to Postgres: {exc}")


def _delete_directory(path: Path) -> None:
    """Delete directory tree safely using FileManager."""
    _checkpoint_file_manager.delete_directory(path, ignore_errors=True)

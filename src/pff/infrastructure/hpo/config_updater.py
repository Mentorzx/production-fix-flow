"""Configuration auto-updater for HPO results.

Automatically applies best hyperparameters from HPO trials to YAML config files.
Automatically applies best hyperparameters from HPO trials to YAML config files.

**Important**: This module ONLY saves raw HPO parameters. Dynamic scaling based
on dataset size happens in the pipeline principal via `pff/utils/ml/adaptive_training.py`.

Design patterns:
- Factory Pattern: Config handler creation based on model type

AGENTS.md Compliance:
- Uses FileManager for I/O operations
- Uses FileManager for I/O operations
- All tunables loaded from config
- Logging follows PT-BR (info/success) / EN (warning/error) contract
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pff.shared import logger
from pff.shared.core.config import DSLFM_CONFIG_PATH
from pff.shared.core.file_manager import FileManager


@dataclass(frozen=True)
class DataScaleProfile:
    """Profile describing dataset scale for reference/logging.

    This is stored alongside HPO results for traceability, but scaling
    decisions are made at runtime by adaptive_training.py in the pipeline.

    Attributes:
        n_entities: Number of unique entities in the KG.
        n_relations: Number of unique relations/predicates.
        n_train_triples: Number of training triples.
        n_valid_triples: Number of validation triples.
        density: Graph density (triples / (entities * relations)).
    """

    n_entities: int = 0
    n_relations: int = 0
    n_train_triples: int = 0
    n_valid_triples: int = 0
    density: float = 0.0

    @classmethod
    def from_data_info(cls, data_info: dict[str, Any]) -> DataScaleProfile:
        """Create profile from HPO data_info dict.

        Args:
            data_info: Dictionary with n_train, n_valid, n_entities, n_predicates.

        Returns:
            DataScaleProfile instance.
        """
        n_entities = int(data_info.get("n_entities", 0))
        n_relations = int(
            data_info.get("n_predicates", data_info.get("n_relations", 0))
        )
        n_train = int(data_info.get("n_train", 0))
        n_valid = int(data_info.get("n_valid", 0))

        possible = max(n_entities * n_relations, 1)
        density = n_train / possible if possible > 0 else 0.0

        return cls(
            n_entities=n_entities,
            n_relations=n_relations,
            n_train_triples=n_train,
            n_valid_triples=n_valid,
            density=density,
        )

    @property
    def scale_tier(self) -> str:
        """Classify dataset into scale tier (for logging/reference only).

        Returns:
            One of: 'tiny', 'small', 'medium', 'large', 'xlarge'
        """
        n = self.n_train_triples
        if n < 10_000:
            return "tiny"
        elif n < 100_000:
            return "small"
        elif n < 1_000_000:
            return "medium"
        elif n < 10_000_000:
            return "large"
        else:
            return "xlarge"


def _load_or_init_config(
    file_manager: FileManager, config_path: Path
) -> dict[str, Any]:
    """Execute load or init config.



    Args:

        file_manager: Input value used by this callable.

        config_path: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if not file_manager.exists(config_path):
        logger.warning(f"Config file not found: {config_path}; creating new config")
        return {"model": {}, "training": {}, "device": {}}
    return file_manager.read(config_path, return_native=True)


def _default_param_mapping() -> dict[str, tuple[str, str]]:
    return {
        "embedding_dim": ("model", "embedding_dim"),
        "batch_size": ("training", "batch_size"),
        "learning_rate": ("training", "learning_rate"),
        "lr": ("training", "learning_rate"),
        "negative_samples": ("training", "negative_samples"),
        "epochs": ("training", "epochs"),
        "adversarial_temperature": ("training", "adversarial_temperature"),
        "lambda_logic": ("logic", "lambda_logic"),
        "lambda_pc": ("pc", "lambda_pc"),
    }


def _apply_param_mapping(
    *,
    best_params: dict[str, Any],
    config: dict[str, Any],
    param_mapping: dict[str, tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Execute apply param mapping.



    Args:

        best_params: Input value used by this callable.

        config: Input value used by this callable.

        param_mapping: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    original_values: dict[str, Any] = {}
    changes: dict[str, dict[str, Any]] = {}
    for param_name, value in best_params.items():
        mapping = param_mapping.get(param_name)
        if mapping is None:
            continue
        section, key = mapping
        config.setdefault(section, {})
        old_value = config[section].get(key)
        original_values[f"{section}.{key}"] = old_value
        if old_value != value:
            config[section][key] = value
            changes[f"{section}.{key}"] = {"old": old_value, "new": value}
    return original_values, changes


def _collect_updated_values(
    config: dict[str, Any],
    param_mapping: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    """Execute collect updated values.



    Args:

        config: Input value used by this callable.

        param_mapping: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    updated: dict[str, Any] = {}
    for section, key in param_mapping.values():
        value = config.get(section, {}).get(key)
        if value is not None:
            updated[f"{section}.{key}"] = value
    return updated


def update_dslfm_config(
    best_params: dict[str, Any],
    config_path: Path | None = None,
    data_profile: DataScaleProfile | None = None,
    file_manager: FileManager | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Update DSLFM config YAML with best HPO parameters.


    Saves raw HPO parameters WITHOUT any scaling - scaling is handled
    at runtime by the pipeline principal via adaptive_training.py.

    Args:
        best_params: Best parameters from HPO (saved as-is).
        config_path: Path to dslfm.yaml (default: config/models/dslfm.yaml).
        data_profile: Dataset profile for reference/logging (NOT used for scaling).
        file_manager: FileManager instance for I/O.
        dry_run: If True, return changes without writing to file.

    Returns:
        Dictionary with 'original', 'updated', 'changes', and metadata keys.
    """
    fm = file_manager or FileManager()
    config_path = config_path or DSLFM_CONFIG_PATH
    config = _load_or_init_config(fm, config_path)
    param_mapping = _default_param_mapping()
    original_values, changes = _apply_param_mapping(
        best_params=best_params,
        config=config,
        param_mapping=param_mapping,
    )

    result = {
        "original": original_values,
        "updated": _collect_updated_values(config, param_mapping),
        "changes": changes,
        "config_path": str(config_path),
    }

    if data_profile is not None:
        result["hpo_data_profile"] = {
            "n_entities": data_profile.n_entities,
            "n_relations": data_profile.n_relations,
            "n_train_triples": data_profile.n_train_triples,
            "n_valid_triples": data_profile.n_valid_triples,
            "scale_tier": data_profile.scale_tier,
        }
        logger.info(
            f"Perfil dos dados HPO registrado: tier={data_profile.scale_tier}, "
            f"triplas={data_profile.n_train_triples:,}"
        )

    if not dry_run and changes:
        fm.save(config, config_path)
        logger.success(f"Configuracao DSLFM atualizada: {config_path}")
        logger.info(f"Parametros alterados: {len(changes)}")
        for path, change in changes.items():
            logger.info(f"  {path}: {change['old']} -> {change['new']}")
    elif dry_run and changes:
        logger.info("Modo dry-run: alteracoes nao salvas")
        for path, change in changes.items():
            logger.info(f"  [DRY-RUN] {path}: {change['old']} -> {change['new']}")
    else:
        logger.info("Nenhuma alteracao necessaria na configuracao")

    return result


def create_config_update_callback(
    config_path: Path | None = None,
    file_manager: FileManager | None = None,
):
    """Create an Optuna callback that updates config on study completion.

    The callback saves raw HPO parameters to the config file. No scaling
    is applied - that's the responsibility of the pipeline principal.

    Args:
        config_path: Path to dslfm.yaml config file.
        file_manager: FileManager instance.

    Returns:
        Callable suitable for Optuna study.optimize(callbacks=[...]).
    """
    fm = file_manager or FileManager()

    class ConfigUpdateCallback:
        """Optuna callback to update config with best parameters.

        Saves raw HPO parameters without any scaling. Dynamic parameter
        adjustment based on dataset size is handled at runtime by
        adaptive_training.py in the pipeline principal.
        """

        def __init__(self):
            """Execute init."""

            self.best_value: float = float("-inf")
            self.best_params: dict[str, Any] = {}
            self._update_applied = False

        def __call__(self, study, trial) -> None:
            """Track best trial during optimization."""
            value = getattr(trial, "value", None)
            if value is None:
                return

            try:
                numeric_value = float(value)
            except Exception:
                return

            if numeric_value > self.best_value:
                self.best_value = numeric_value
                self.best_params = dict(getattr(trial, "params", {}))

        def finalize(self, data_info: dict[str, Any] | None = None) -> dict[str, Any]:
            """Apply final config update after optimization completes.

            Args:
                data_info: Optional data info for reference profile logging (no scaling).

            Returns:
                Update result dictionary.
            """
            if self._update_applied:
                logger.info("Atualizacao de config ja aplicada")
                return {"skipped": True}

            if not self.best_params:
                logger.warning("No best params to apply; skipping config update")
                return {"skipped": True, "reason": "no_params"}

            data_profile = None
            if data_info:
                data_profile = DataScaleProfile.from_data_info(data_info)

            result = update_dslfm_config(
                best_params=self.best_params,
                config_path=config_path,
                data_profile=data_profile,
                file_manager=fm,
                dry_run=False,
            )

            self._update_applied = True
            return result

    return ConfigUpdateCallback()


def _sanitize_for_json(value: Any) -> Any:
    """Recursively sanitize values to JSON-serializable types.

    Args:
        value: Any Python value.

    Returns:
        JSON-safe representation (primitives, lists, dicts).
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    type_name = type(value).__name__
    if type_name in ("DataFrame", "LazyFrame", "Series"):
        return f"<{type_name}: {getattr(value, 'shape', 'unknown')}>"
    if type_name == "Study":
        return f"<Optuna Study: {getattr(value, 'study_name', 'unknown')}>"
    try:
        return float(value)
    except (TypeError, ValueError):
        return f"<{type_name}>"


def export_hpo_summary(
    result: dict[str, Any],
    output_dir: Path,
    file_manager: FileManager | None = None,
) -> Path:
    """Export HPO summary with best parameters.

    Args:
        result: HPO optimization result dictionary.
        output_dir: Directory to save summary.
        file_manager: FileManager instance.

    Returns:
        Path to saved summary file.
    """
    fm = file_manager or FileManager()

    summary = {
        "best_params": _sanitize_for_json(result.get("best_params", {})),
        "best_value": _sanitize_for_json(result.get("best_value")),
        "n_trials": _sanitize_for_json(result.get("n_trials", 0)),
        "optimization_time": _sanitize_for_json(result.get("optimization_time", 0)),
        "data_info": _sanitize_for_json(result.get("real_data_info", {})),
    }

    if "real_data_info" in result:
        ref_profile = DataScaleProfile.from_data_info(result["real_data_info"])
        summary["hpo_data_profile"] = {
            "n_entities": ref_profile.n_entities,
            "n_relations": ref_profile.n_relations,
            "n_train_triples": ref_profile.n_train_triples,
            "scale_tier": ref_profile.scale_tier,
            "note": "Dynamic scaling handled by adaptive_training.py at runtime",
        }

    output_path = output_dir / "hpo_summary.json"
    fm.save(summary, output_path)
    logger.info(f"Sumario HPO exportado: {output_path}")

    return output_path

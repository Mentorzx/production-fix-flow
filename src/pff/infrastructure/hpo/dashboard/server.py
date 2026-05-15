"""Peak State HPO Dashboard Server.

Implements:
1. Strict Cache-Control:
    - index.html: no-cache, must-revalidate
    - /dist/*: public, max-age=31536000, immutable
2. Multi-path Data Loader: fallbacks between cache/ and outputs/
3. Lookback Logic: holds last valid validation metrics if current epoch is train-only.
4. PID Watchdog: self-terminates if parent dies.
"""

import http.server
import os
import signal
import socketserver
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlparse

from ruamel.yaml.comments import CommentedSeq

from pff.infrastructure.hpo.config_loader import (
    clear_config_cache,
    load_live_plot_settings,
)
from pff.infrastructure.hpo.dashboard.exporters import (
    EXPORT_HANDLERS as _EXPORT_HANDLERS,
)
from pff.infrastructure.hpo.dashboard.exporters import (
    export_json as _export_json,
)
from pff.infrastructure.hpo.dashboard.exporters import (
    normalize_direction_label as _normalize_direction_label,
)
from pff.infrastructure.hpo.dashboard.log_parsing import (
    MAX_LOG_ENTRIES as _MAX_LOG_ENTRIES,
)
from pff.infrastructure.hpo.dashboard.log_parsing import (
    MAX_TAIL_BYTES as _MAX_TAIL_BYTES,
)
from pff.infrastructure.hpo.dashboard.log_parsing import (
    MAX_TAIL_LINES as _MAX_TAIL_LINES,
)
from pff.infrastructure.hpo.dashboard.log_parsing import (
    load_json_payload as _load_json_payload,
)
from pff.infrastructure.hpo.dashboard.log_parsing import (
    normalize_log_entries as _normalize_log_entries,
)
from pff.infrastructure.hpo.dashboard.log_parsing import (
    read_tail_lines as _read_tail_lines,
)
from pff.infrastructure.hpo.search_space_advisor import (
    ADVISOR_VERSION,
    SearchSpaceAdvisor,
    compute_dataset_profile_fingerprint,
    generate_search_space_patch,
)
from pff.shared.acceleration.concurrency import HardwareManager
from pff.shared.core.cache import CacheManager
from pff.shared.core.config import OPTIMIZATION_CONFIG_PATH, settings
from pff.shared.core.file_manager import FileManager
from pff.shared.core.file_manager.handlers.yaml import YAMLHandler
from pff.shared.core.logging import LOG_DIR, create_isolated_logger, logger
from pff.shared.ops.global_interrupt_manager import get_interrupt_manager, should_stop

DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_DIR = DASHBOARD_DIR / "static"
DIST_DIR = DASHBOARD_DIR / "dist"
BASE_DIR = settings.ROOT_DIR
_DASHBOARD_LOGGER = None
DATA_CACHE_PATH: Path | None = None
_DASHBOARD_RUNTIME_CACHE = CacheManager(max_memory_items=256)
_CACHE_KEY_DATA_PATHS = "hpo_dashboard:data_paths"
_CACHE_KEY_TELEMETRY = "hpo_dashboard:telemetry"
_CACHE_KEY_LOOKBACK = "hpo_dashboard:lookback"
_CACHE_KEY_DATA_SOURCE = "hpo_dashboard:data_source"
_DATA_PATHS_CACHE_TTL_S = 1.0
_TELEMETRY_CACHE_TTL_S = 1.0
_HARDWARE_HISTORY: dict[str, Any] = {"items": [], "last_id": 0}
_SEARCH_SPACE_ADVISOR: SearchSpaceAdvisor | None = None
_MAX_ADVISOR_BOUND_EXPANSION_FACTOR = 10.0
_LOOKBACK_LOCK = threading.Lock()
_LOOKBACK_DEFAULT: dict[str, Any] = {
    "gen_gap": None,
    "confusion_matrix": None,
    "confusion_matrices": None,
    "last_valid_epoch": -1,
    "source_trial": -1,
    "live_best_metrics": {},
}
_LIVE_STATUS_MAX_AGE_SECONDS = 900.0
_DATA_SOURCE_RECENCY_WINDOW_S = 5.0


def _get_lookback_memory() -> dict[str, Any]:
    payload = _DASHBOARD_RUNTIME_CACHE.get(_CACHE_KEY_LOOKBACK)
    if isinstance(payload, dict):
        return payload
    seeded = dict(_LOOKBACK_DEFAULT)
    _DASHBOARD_RUNTIME_CACHE.set(_CACHE_KEY_LOOKBACK, seeded)
    return seeded


def _set_lookback_memory(payload: dict[str, Any]) -> None:
    _DASHBOARD_RUNTIME_CACHE.set(_CACHE_KEY_LOOKBACK, payload)


def _get_search_space_advisor() -> SearchSpaceAdvisor:
    global _SEARCH_SPACE_ADVISOR
    if _SEARCH_SPACE_ADVISOR is None:
        _SEARCH_SPACE_ADVISOR = SearchSpaceAdvisor()
    return _SEARCH_SPACE_ADVISOR


def _query_flag(path: str, key: str) -> bool:
    try:
        query = parse_qs(urlparse(path).query, keep_blank_values=True)
    except Exception:
        return False
    values = query.get(key, [])
    for value in values:
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
    return False


def _get_dashboard_logger():
    """Execute get dashboard logger.



    Returns:

        Return value produced by the callable.

    """

    global _DASHBOARD_LOGGER
    if _DASHBOARD_LOGGER is None:
        _DASHBOARD_LOGGER = create_isolated_logger("hpo_dashboard", log_dir=LOG_DIR / "dashboard")
    return _DASHBOARD_LOGGER


def _resolve_dashboard_data_path(live_cfg: dict[str, Any] | None = None) -> Path:
    """Execute resolve dashboard data path.



    Args:

        live_cfg: Optional input value.



    Returns:

        Return value produced by the callable.

    """

    if DATA_CACHE_PATH is not None:
        return DATA_CACHE_PATH
    cfg = live_cfg or load_live_plot_settings()
    data_path = cfg.get("dashboard_data_path")
    if data_path:
        resolved = Path(data_path)
        if not resolved.is_absolute():
            resolved = settings.ROOT_DIR / resolved
        return resolved
    return settings.CACHE_DIR / "hpo" / "dashboard_data.json"


def _reset_dashboard_paths_cache() -> None:
    """Execute reset dashboard paths cache."""
    _DASHBOARD_RUNTIME_CACHE.invalidate(pattern=f"^{_CACHE_KEY_DATA_PATHS}$")


def _reset_telemetry_cache() -> None:
    """Execute reset telemetry cache."""
    _DASHBOARD_RUNTIME_CACHE.invalidate(pattern=f"^{_CACHE_KEY_TELEMETRY}$")


def _append_hardware_history(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute append hardware history.



    Args:

        telemetry: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    cpu = telemetry.get("cpu_usage")
    if cpu is None:
        cpu = telemetry.get("cpu_utilization")
    ram = telemetry.get("ram_usage_pct")
    if ram is None:
        ram = telemetry.get("ram_utilization")
    gpu_util = None
    vram_util = None
    gpus = telemetry.get("gpus")
    if isinstance(gpus, list) and gpus:
        gpu0 = gpus[0]
        if isinstance(gpu0, dict):
            gpu_util = gpu0.get("utilization_total")
            if gpu_util is None:
                gpu_util = gpu0.get("utilization")
            vram_util = gpu0.get("vram_usage_pct")
    if gpu_util is None:
        gpu_util = telemetry.get("gpu_utilization")
    if vram_util is None:
        vram_util = telemetry.get("vram_utilization")

    if cpu is None and ram is None and gpu_util is None:
        return cast(list[dict[str, Any]], _HARDWARE_HISTORY["items"])

    _HARDWARE_HISTORY["last_id"] += 1
    sample = {
        "id": _HARDWARE_HISTORY["last_id"],
        "cpu_usage": float(cpu) if cpu is not None else None,
        "ram_usage_pct": float(ram) if ram is not None else None,
        "gpu_utilization": float(gpu_util) if gpu_util is not None else None,
        "vram_usage_pct": float(vram_util) if vram_util is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    items: list[dict[str, Any]] = _HARDWARE_HISTORY["items"]
    items.append(sample)
    if len(items) > 180:
        _HARDWARE_HISTORY["items"] = items[-180:]
    return cast(list[dict[str, Any]], _HARDWARE_HISTORY["items"])


def _collect_dashboard_data_paths() -> list[Path]:
    """Execute collect dashboard data paths.



    Returns:

        Return value produced by the callable.

    """

    now = time.time()
    cached_entry = _DASHBOARD_RUNTIME_CACHE.get(_CACHE_KEY_DATA_PATHS)
    cached_paths: tuple[Path, ...] = tuple()
    if isinstance(cached_entry, dict):
        cached_raw_paths = cached_entry.get("paths", tuple())
        if isinstance(cached_raw_paths, tuple):
            cached_paths = tuple(path for path in cached_raw_paths if isinstance(path, Path))
        cached_last_refresh = float(cached_entry.get("last_refresh", 0.0) or 0.0)
        if cached_paths and now - cached_last_refresh < _DATA_PATHS_CACHE_TTL_S:
            return list(cached_paths)
    else:
        cached_entry = {}

    live_cfg = load_live_plot_settings()
    data_cache_path = _resolve_dashboard_data_path(live_cfg)
    output_subdir = live_cfg.get("output_subdir", "optimization/plots")
    explicit_data_path = bool(live_cfg.get("dashboard_data_path"))
    live_plot_dir = settings.OUTPUTS_DIR / Path(output_subdir)
    cache_root = settings.CACHE_DIR / "hpo"
    cache_root_mtime = cache_root.stat().st_mtime if cache_root.exists() else None

    if (
        cached_paths
        and cached_entry.get("output_subdir") == output_subdir
        and cached_entry.get("data_cache_path") == data_cache_path
        and bool(cached_entry.get("explicit_data_path")) == explicit_data_path
        and cached_entry.get("cache_root_mtime") == cache_root_mtime
    ):
        _DASHBOARD_RUNTIME_CACHE.set(
            _CACHE_KEY_DATA_PATHS,
            {**cached_entry, "last_refresh": now},
        )
        return list(cached_paths)

    candidates = [data_cache_path]
    if not explicit_data_path:
        candidates.insert(0, BASE_DIR / "outputs" / "dashboard_data.json")
    else:
        candidates.append(BASE_DIR / "outputs" / "dashboard_data.json")

    candidates.extend(
        [
            BASE_DIR / ".cache" / "hpo" / "dashboard_data.json",
            DASHBOARD_DIR / "dashboard_data.json",
            live_plot_dir / "dashboard_data.json",
        ]
    )

    if cache_root.exists():
        candidates.extend(list(cache_root.rglob("dashboard_data.json")))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    _DASHBOARD_RUNTIME_CACHE.set(
        _CACHE_KEY_DATA_PATHS,
        {
            "paths": tuple(unique),
            "last_refresh": now,
            "output_subdir": output_subdir,
            "data_cache_path": data_cache_path,
            "explicit_data_path": explicit_data_path,
            "cache_root_mtime": cache_root_mtime,
        },
    )
    return unique


def _log_event(
    level: str,
    message: str,
    *,
    key_parameters: dict[str, Any] | None = None,
    stop_reason: str = "none",
) -> None:
    """Execute log event.



    Args:

        level: Input value used by this callable.

        message: Input value used by this callable.

        key_parameters: Optional input value.

        stop_reason: Optional input value.

    """

    bound = logger.bind(
        component="hpo_dashboard",
        key_parameters=key_parameters or {},
        stop_reason=stop_reason,
    )
    getattr(bound, level)(message)


def _get_cached_telemetry(hardware_manager: HardwareManager) -> dict[str, Any]:
    """Execute get cached telemetry.



    Args:

        hardware_manager: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    cached = _DASHBOARD_RUNTIME_CACHE.get(_CACHE_KEY_TELEMETRY)
    if isinstance(cached, dict):
        return cast(dict[str, Any], cached)
    telemetry: dict[str, Any] = hardware_manager.get_telemetry()
    _DASHBOARD_RUNTIME_CACHE.set(
        _CACHE_KEY_TELEMETRY,
        telemetry,
        ttl=max(1, int(round(_TELEMETRY_CACHE_TTL_S))),
    )
    return telemetry


_LOGS_DIR = BASE_DIR / "logs" / "readable"

_METRIC_KEYS = (
    "mrr",
    "mcc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "pr_auc",
    "hits@1",
    "hits@3",
    "hits@10",
    "hits1",
    "hits3",
    "hits10",
)


def _epoch_has_validation_signals(item: dict[str, Any]) -> bool:
    """Return True when epoch metrics include validation/evaluation indicators."""
    eval_keys = (
        "mrr",
        "mcc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "pr_auc",
        "hits@1",
        "hits@3",
        "hits@10",
        "hits1",
        "hits3",
        "hits10",
        "tp",
        "vp",
        "tn",
        "vn",
        "fp",
        "fn",
        "decision_threshold",
    )
    return any(item.get(key) is not None for key in eval_keys)


def _normalize_live_epoch_losses(live_status: dict[str, Any]) -> None:
    """Normalize epoch loss aliases for dashboard consumers."""
    history = live_status.get("epoch_history")
    if not isinstance(history, list):
        return
    for row in history:
        if not isinstance(row, dict):
            continue
        if row.get("train_loss") is None:
            row["train_loss"] = (
                row.get("loss") or row.get("binary_loss") or row.get("train_binary_loss")
            )
        if row.get("val_loss") is None:
            row["val_loss"] = (
                row.get("eval_loss") or row.get("val_binary_loss") or row.get("test_loss")
            )
        if row.get("val_loss") is None and _epoch_has_validation_signals(row):
            row["val_loss"] = row.get("binary_loss")
        if row.get("loss") is None:
            row["loss"] = row.get("train_loss") or row.get("val_loss") or row.get("binary_loss")


def _epoch_score(item: dict[str, Any]) -> float:
    """Extracts the best available score from an epoch metrics dict."""
    raw = item.get("score") or item.get("mrr") or item.get("mcc") or item.get("accuracy") or 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _has_metrics(item: dict[str, Any]) -> bool:
    """Returns True if the epoch dict contains at least one non-zero metric."""
    for key in _METRIC_KEYS:
        val = item.get(key)
        if isinstance(val, (int, float)) and val != 0:
            return True
    return False


def _clean_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Filters out None values from a metrics dict."""
    return {k: v for k, v in payload.items() if v is not None}


def _normalize_study_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _parse_iso_ts(value: Any) -> float:
    if not isinstance(value, str):
        return 0.0
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return 0.0


def _trial_id_from_live_status(payload: dict[str, Any]) -> int | None:
    raw_trial = payload.get("trial_number")
    if raw_trial is None:
        return None
    try:
        base = int(raw_trial)
    except (TypeError, ValueError):
        return None
    if base < 0:
        return None
    return base + 1


def _payload_live_trial_id(payload: dict[str, Any]) -> int | None:
    live_status = payload.get("liveStatus")
    if not isinstance(live_status, dict):
        return None
    return _trial_id_from_live_status(live_status)


def _extract_active_trial_ids(raw_data: dict[str, Any]) -> set[int]:
    trials = raw_data.get("trials")
    if not isinstance(trials, list):
        return set()
    active: set[int] = set()
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        state = str(trial.get("state", "")).upper()
        if state not in {"RUNNING", "WAITING"}:
            continue
        raw_trial_id = trial.get("id")
        if raw_trial_id is None:
            continue
        try:
            trial_id = int(raw_trial_id)
        except (TypeError, ValueError):
            continue
        if trial_id > 0:
            active.add(trial_id)
    return active


def _coerce_positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _coerce_non_negative_int(value: Any, *, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed >= 0 else int(default)


def _payload_updated_ts(payload: dict[str, Any], path: Path) -> float:
    raw_updated = payload.get("updatedAt")
    if isinstance(raw_updated, str):
        normalized = raw_updated.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            pass
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _load_raw_dashboard_data(
    active_study_name: str | None = None,
    *,
    preferred_live_trial_id: int | None = None,
) -> dict[str, Any]:
    """Reads dashboard JSON and selects the most recent valid payload.

    When ``active_study_name`` is available, matching study payloads are preferred.
    """
    paths = _collect_dashboard_data_paths()
    existing_files = [p for p in paths if FileManager.exists(p)]
    if not existing_files:
        return {}
    priority_by_path = {path: idx for idx, path in enumerate(paths)}
    payloads: list[tuple[dict[str, Any], Path, float, int, int | None]] = []
    for candidate in existing_files:
        try:
            payload = _load_json_payload(FileManager.read_bytes(candidate))
        except Exception as e:
            _log_event(
                "warning",
                f"Failed to load dashboard data: {e}",
                key_parameters={"path": str(candidate)},
                stop_reason="dashboard_data_read_failed",
            )
            continue
        payloads.append(
            (
                payload,
                candidate,
                _payload_updated_ts(payload, candidate),
                int(priority_by_path.get(candidate, 10**6)),
                _payload_live_trial_id(payload),
            )
        )
    if not payloads:
        return {}

    normalized_active_study = _normalize_study_name(active_study_name)
    if normalized_active_study:
        matched_study = [
            entry
            for entry in payloads
            if _normalize_study_name(entry[0].get("studyName")) == normalized_active_study
        ]
        if matched_study:
            payloads = matched_study

    if preferred_live_trial_id is not None:
        matched_live_trial = [
            entry for entry in payloads if entry[4] == int(preferred_live_trial_id)
        ]
        if matched_live_trial:
            payloads = matched_live_trial

    freshest_ts = max(entry[2] for entry in payloads)
    recent_cutoff = freshest_ts - _DATA_SOURCE_RECENCY_WINDOW_S
    recent_payloads = [entry for entry in payloads if entry[2] >= recent_cutoff]
    selection_pool = recent_payloads if recent_payloads else payloads

    has_explicit_selection_hint = bool(normalized_active_study) or (
        preferred_live_trial_id is not None
    )
    if has_explicit_selection_hint:
        cached_source = _DASHBOARD_RUNTIME_CACHE.get(_CACHE_KEY_DATA_SOURCE)
        if (
            isinstance(cached_source, dict)
            and cached_source.get("study") == normalized_active_study
            and isinstance(cached_source.get("path"), str)
        ):
            cached_path = Path(cached_source["path"])
            cached_candidates = [entry for entry in selection_pool if entry[1] == cached_path]
            if cached_candidates:
                return max(cached_candidates, key=lambda entry: entry[2])[0]

        best_priority = min(entry[3] for entry in selection_pool)
        prioritized = [entry for entry in selection_pool if entry[3] == best_priority]
        selected = max(prioritized, key=lambda entry: entry[2])
    else:
        selected = max(selection_pool, key=lambda entry: entry[2])

    _DASHBOARD_RUNTIME_CACHE.set(
        _CACHE_KEY_DATA_SOURCE,
        {
            "study": normalized_active_study,
            "path": str(selected[1]),
            "updated_ts": float(selected[2]),
        },
    )
    return selected[0]


def _refresh_trial_count_fields(raw_data: dict[str, Any]) -> None:
    """Normalizes count-related fields used by dashboard cards and ETA widgets."""
    hpo_defaults = settings.HPO_CONFIG.get("defaults", {})
    default_total_trials = _coerce_positive_int(hpo_defaults.get("n_trials", 50), default=50)

    total_target = _coerce_positive_int(
        raw_data.get("total_trials_target", raw_data.get("totalTrials")),
        default=default_total_trials,
    )
    raw_data["total_trials_target"] = total_target
    raw_data["totalTrials"] = total_target

    trials = raw_data.get("trials")
    if not isinstance(trials, list):
        raw_data["completed_trials_all"] = _coerce_non_negative_int(
            raw_data.get("completed_trials_all"), default=0
        )
        raw_data["completed_trials_non_warmstart"] = _coerce_non_negative_int(
            raw_data.get("completed_trials_non_warmstart"), default=0
        )
        raw_data["warmstart_trials"] = _coerce_non_negative_int(
            raw_data.get("warmstart_trials"), default=0
        )
        return

    completed_trials_all = 0
    completed_trials_non_warmstart = 0
    warmstart_trials = 0
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        is_warmstart = bool(trial.get("warmstart"))
        if is_warmstart:
            warmstart_trials += 1
        if str(trial.get("state")) == "COMPLETE":
            completed_trials_all += 1
            if not is_warmstart:
                completed_trials_non_warmstart += 1

    raw_data["completed_trials_all"] = completed_trials_all
    raw_data["completed_trials_non_warmstart"] = completed_trials_non_warmstart
    raw_data["warmstart_trials"] = warmstart_trials


def _has_usable_search_space_advice(payload: Any) -> bool:
    """Return whether cached search space advice is usable for UI rendering."""
    if not isinstance(payload, dict):
        return False
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        advisor_version = metadata.get("advisor_version")
        if advisor_version is not None and advisor_version != ADVISOR_VERSION:
            return False
    recommendations = payload.get("recommendations")
    if isinstance(recommendations, list) and len(recommendations) > 0:
        return True
    if isinstance(metadata, dict) and metadata.get("insufficient_evidence") is True:
        return True
    return False


_SEARCH_SPACE_PARAM_MAP: dict[str, dict[str, tuple[str, ...]]] = {
    "learning_rate": {
        "low": ("dslfm_kgc", "training", "lr_low"),
        "high": ("dslfm_kgc", "training", "lr_high"),
    },
    "batch_size": {
        "low": ("dslfm_kgc", "training", "batch_size_low"),
        "high": ("dslfm_kgc", "training", "batch_size_high"),
    },
    "negative_sample_size": {
        "low": ("dslfm_kgc", "training", "negative_sample_size_low"),
        "high": ("dslfm_kgc", "training", "negative_sample_size_high"),
    },
    "dslfm_epochs": {
        "low": ("dslfm_kgc", "training", "epochs_low"),
        "high": ("dslfm_kgc", "training", "epochs_high"),
    },
    "contrastive_temperature": {
        "low": ("dslfm_kgc", "contrastive", "temperature_low"),
        "high": ("dslfm_kgc", "contrastive", "temperature_high"),
    },
    "num_global_negatives": {
        "low": ("dslfm_kgc", "contrastive", "num_global_negatives_low"),
        "high": ("dslfm_kgc", "contrastive", "num_global_negatives_high"),
    },
    "adversarial_temperature": {
        "low": ("dslfm_kgc", "sampling", "adv_temperature_low"),
        "high": ("dslfm_kgc", "sampling", "adv_temperature_high"),
    },
    "lambda_logic": {
        "low": ("dslfm_kgc", "logic", "lambda_logic_low"),
        "high": ("dslfm_kgc", "logic", "lambda_logic_high"),
    },
    "t_norm": {"choices": ("dslfm_kgc", "logic", "t_norm_choices")},
    "attr_hidden_dim": {"choices": ("dslfm_kgc", "architecture", "hidden_dim_choices")},
    "embedding_dim": {"choices": ("dslfm_kgc", "architecture", "feature_dim_choices")},
    "max_communities": {"choices": ("dslfm_kgc", "architecture", "max_communities_choices")},
    "ibp_alpha": {
        "low": ("dslfm_kgc", "architecture", "ibp_alpha_low"),
        "high": ("dslfm_kgc", "architecture", "ibp_alpha_high"),
    },
    "kl_weight": {
        "low": ("dslfm_kgc", "architecture", "kl_weight_low"),
        "high": ("dslfm_kgc", "architecture", "kl_weight_high"),
    },
    "lambda_pc": {
        "low": ("dslfm_kgc", "pc", "lambda_pc_low"),
        "high": ("dslfm_kgc", "pc", "lambda_pc_high"),
    },
    "pruning_threshold": {
        "low": ("dslfm_kgc", "pc", "pruning_threshold_low"),
        "high": ("dslfm_kgc", "pc", "pruning_threshold_high"),
    },
    "rebuild_every": {
        "low": ("dslfm_kgc", "pc", "rebuild_every_low"),
        "high": ("dslfm_kgc", "pc", "rebuild_every_high"),
    },
    "max_circuit_depth": {"choices": ("dslfm_kgc", "pc", "depth_choices")},
    "validate_every": {
        "low": ("adaptive_range_factors", "validate_every_low"),
        "high": ("adaptive_range_factors", "validate_every_high"),
    },
    "early_stopping_patience": {
        "low": ("adaptive_range_factors", "early_stopping_patience_low"),
        "high": ("adaptive_range_factors", "early_stopping_patience_high"),
    },
    "min_delta": {
        "low": ("adaptive_range_factors", "min_delta_low"),
        "high": ("adaptive_range_factors", "min_delta_high"),
    },
}


def _ensure_nested_config(config: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    cursor = config
    for key in keys[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            cursor[key] = {}
        cursor = cursor[key]
    return cursor


def _load_optimization_config_rt() -> dict[str, Any]:
    config = YAMLHandler().read(Path(OPTIMIZATION_CONFIG_PATH))
    if not isinstance(config, dict):
        raise ValueError(f"Optimization config at {OPTIMIZATION_CONFIG_PATH} must be a mapping")
    return config


def _cast_numeric_bound(value: Any, entry: dict[str, Any]) -> Any:
    if value is None:
        return None
    entry_type = entry.get("type")
    if entry_type == "int":
        return int(round(float(value)))
    if isinstance(value, str):
        return float(value)
    if isinstance(value, int):
        return float(value)
    return value


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, Real):
        return False
    value_float = float(value)
    return value_float == value_float and value_float not in (
        float("inf"),
        float("-inf"),
    )


def _is_numeric_patch_safe(
    *,
    current_low: Any,
    current_high: Any,
    new_low: Any,
    new_high: Any,
) -> bool:
    if not (_is_finite_number(new_low) and _is_finite_number(new_high)):
        return False

    new_low_float = float(new_low)
    new_high_float = float(new_high)
    if new_low_float > new_high_float:
        return False

    if _is_finite_number(current_low):
        current_low_float = float(current_low)
        if current_low_float >= 0.0 and new_low_float < 0.0:
            return False
        if (
            current_low_float > 0.0
            and new_low_float > 0.0
            and (current_low_float / new_low_float) > _MAX_ADVISOR_BOUND_EXPANSION_FACTOR
        ):
            return False

    if _is_finite_number(current_high):
        current_high_float = float(current_high)
        if (
            current_high_float > 0.0
            and new_high_float > 0.0
            and (new_high_float / current_high_float) > _MAX_ADVISOR_BOUND_EXPANSION_FACTOR
        ):
            return False

    return True


def _normalize_choice_values(values: list[Any], current: Any) -> list[Any]:
    if not isinstance(current, list) or not current:
        return values
    if all(isinstance(item, int) for item in current):
        normalized: list[Any] = []
        for item in values:
            if isinstance(item, str) and item.isdigit():
                normalized.append(int(item))
            else:
                normalized.append(item)
        return normalized
    if all(isinstance(item, float) for item in current):
        normalized = []
        for item in values:
            normalized.append(float(item))
        return normalized
    return values


def _wrap_sequence(values: list[Any], current: Any) -> list[Any] | CommentedSeq:
    if isinstance(current, CommentedSeq):
        seq = CommentedSeq(values)
        if current.fa.flow_style():
            seq.fa.set_flow_style()
        else:
            seq.fa.set_block_style()
        return seq
    return values


def _extract_patch_bounds(entry: dict[str, Any]) -> tuple[float, float] | None:
    if entry.get("type") == "fixed":
        value = entry.get("value")
        if value is None:
            return None
        return float(value), float(value)
    low = entry.get("low", entry.get("new_low"))
    high = entry.get("high", entry.get("new_high"))
    if low is None or high is None:
        return None
    return float(low), float(high)


def _extract_patch_choices(entry: dict[str, Any]) -> list[Any] | None:
    if entry.get("type") == "fixed":
        value = entry.get("value")
        return [value] if value is not None else None
    choices = entry.get("choices")
    if choices is None:
        return None
    if isinstance(choices, list):
        return choices
    return [choices]


def _apply_search_space_patch_to_config(
    config: dict[str, Any],
    patch: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    applied: list[str] = []
    skipped: list[str] = []
    for param, entry in patch.items():
        mapping = _SEARCH_SPACE_PARAM_MAP.get(param)
        if mapping is None or not isinstance(entry, dict):
            skipped.append(param)
            continue
        if "choices" in mapping:
            choices = _extract_patch_choices(entry)
            if choices is None:
                skipped.append(param)
                continue
            cursor = _ensure_nested_config(config, mapping["choices"])
            key = mapping["choices"][-1]
            current_value = cursor.get(key)
            normalized = _normalize_choice_values(choices, current_value)
            updated = _wrap_sequence(normalized, current_value)
            if current_value == updated:
                skipped.append(param)
                continue
            cursor[key] = updated
        else:
            bounds = _extract_patch_bounds(entry)
            if bounds is None:
                skipped.append(param)
                continue
            low_value, high_value = bounds
            low_value = _cast_numeric_bound(low_value, entry)
            high_value = _cast_numeric_bound(high_value, entry)
            low_path = mapping["low"]
            high_path = mapping["high"]
            low_cursor = _ensure_nested_config(config, low_path)
            high_cursor = _ensure_nested_config(config, high_path)
            current_low = low_cursor.get(low_path[-1])
            current_high = high_cursor.get(high_path[-1])
            if not _is_numeric_patch_safe(
                current_low=current_low,
                current_high=current_high,
                new_low=low_value,
                new_high=high_value,
            ):
                skipped.append(param)
                continue
            if current_low == low_value and current_high == high_value:
                skipped.append(param)
                continue
            low_cursor[low_path[-1]] = low_value
            high_cursor[high_path[-1]] = high_value
        applied.append(param)
    return config, applied, skipped


def _apply_patch_to_search_space(
    search_space: dict[str, Any],
    patch: dict[str, Any],
) -> None:
    for param, entry in patch.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "categorical":
            search_space[param] = {
                "type": "categorical",
                "choices": entry.get("choices", []),
            }
            continue
        if entry.get("type") == "fixed":
            search_space[param] = {"type": "fixed", "value": entry.get("value")}
            continue
        existing = search_space.get(param)
        base = existing if isinstance(existing, dict) else {}
        updated = {**base}
        low = entry.get("low", entry.get("new_low"))
        high = entry.get("high", entry.get("new_high"))
        if low is not None:
            updated["low"] = low
        if high is not None:
            updated["high"] = high
        search_space[param] = updated


def _mark_search_space_advice_applied(payload: dict[str, Any], applied_params: list[str]) -> None:
    advice = payload.get("searchSpaceAdvice")
    if not isinstance(advice, dict):
        return
    recommendations = advice.get("recommendations")
    if isinstance(recommendations, list):
        advice["recommendations"] = [
            rec
            for rec in recommendations
            if isinstance(rec, dict) and rec.get("param_name") not in applied_params
        ]
    metadata = advice.setdefault("metadata", {})
    if isinstance(metadata, dict):
        applied_existing = metadata.get("applied_params", [])
        if not isinstance(applied_existing, list):
            applied_existing = []
        merged = sorted({*applied_existing, *applied_params})
        metadata["applied_params"] = merged


def _mark_search_space_advice_ignored(payload: dict[str, Any], ignored_params: list[str]) -> None:
    advice = payload.get("searchSpaceAdvice")
    if not isinstance(advice, dict):
        return
    metadata = advice.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return
    ignored_existing = metadata.get("ignored_params", [])
    if not isinstance(ignored_existing, list):
        ignored_existing = []
    merged = sorted({*ignored_existing, *ignored_params})
    metadata["ignored_params"] = merged
    recommendations = advice.get("recommendations")
    if isinstance(recommendations, list):
        advice["recommendations"] = [
            rec
            for rec in recommendations
            if isinstance(rec, dict) and rec.get("param_name") not in merged
        ]


def _update_dashboard_payloads(
    patch: dict[str, Any],
    applied_params: list[str] | None = None,
    ignored_params: list[str] | None = None,
) -> list[str]:
    updated_paths: list[str] = []
    for path in _collect_dashboard_data_paths():
        if not FileManager.exists(path):
            continue
        try:
            payload = FileManager.read(path, return_native=True)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        search_space = payload.get("searchSpace")
        if isinstance(search_space, dict) and patch:
            _apply_patch_to_search_space(search_space, patch)
            payload["searchSpace"] = search_space
        if applied_params:
            _mark_search_space_advice_applied(payload, applied_params)
        if ignored_params:
            _mark_search_space_advice_ignored(payload, ignored_params)
        try:
            FileManager.save(payload, path)
            updated_paths.append(str(path))
        except Exception:
            continue
    return updated_paths


def _load_live_status(
    *,
    preferred_study_name: str | None = None,
    preferred_trial_ids: set[int] | None = None,
) -> dict[str, Any] | None:
    """Read live status snapshots and select a stable active trial in parallel mode."""
    plots_dir = BASE_DIR / "outputs" / "optimization" / "plots"
    legacy_path = plots_dir / "live_status.json"
    trial_status_dir = plots_dir / "live_status"
    candidate_paths: list[Path] = []
    if trial_status_dir.exists():
        candidate_paths.extend(sorted(trial_status_dir.glob("trial_*.json")))
    if legacy_path.exists():
        candidate_paths.append(legacy_path)

    if not candidate_paths:
        return None

    preferred_study = _normalize_study_name(preferred_study_name)
    now_ts = time.time()
    candidates: list[tuple[dict[str, Any], float]] = []
    for path in candidate_paths:
        try:
            data = _load_json_payload(FileManager.read_bytes(path))
        except Exception:
            continue
        status_study = _normalize_study_name(data.get("study_name") or data.get("studyName"))
        if preferred_study and status_study != preferred_study:
            continue

        updated_ts = _parse_iso_ts(data.get("updated_at"))
        if updated_ts <= 0.0:
            try:
                updated_ts = float(path.stat().st_mtime)
            except OSError:
                updated_ts = 0.0
        if updated_ts > 0.0 and now_ts - updated_ts > _LIVE_STATUS_MAX_AGE_SECONDS:
            # Ignore stale per-trial files from dead/crashed runs.
            continue
        _normalize_live_epoch_losses(data)
        candidates.append((data, updated_ts))

    if not candidates:
        return None

    normalized_preferred_ids = {
        int(trial_id) for trial_id in (preferred_trial_ids or set()) if int(trial_id) > 0
    }
    if normalized_preferred_ids:
        preferred_candidates = [
            entry
            for entry in candidates
            if (_trial_id_from_live_status(entry[0]) or -1) in normalized_preferred_ids
        ]
        if preferred_candidates:
            candidates = preferred_candidates
    candidates.sort(
        key=lambda entry: (
            _trial_id_from_live_status(entry[0]) or 10**9,
            -entry[1],
        )
    )
    return candidates[0][0]


def _collect_terminal_logs(
    live_status: dict[str, Any] | None,
    raw_data: dict[str, Any],
) -> dict[str, Any] | None:
    """Collects and normalizes terminal log entries into live_status."""
    try:
        if not _LOGS_DIR.exists():
            return live_status

        all_candidate_lines: list[str] = []
        date_prefix = datetime.now().strftime("%Y-%m-%d")

        relevant_files = []
        for suffix in ["error.log", "warning.log"]:
            path = _LOGS_DIR / f"{date_prefix}.{suffix}"
            if path.exists():
                relevant_files.append(path)

        if not relevant_files:
            log_files = sorted(
                [
                    f
                    for f in _LOGS_DIR.glob("*.log")
                    if "dashboard" not in f.name and "server" not in f.name
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if log_files:
                relevant_files = [log_files[0]]

        for path in relevant_files:
            chunk = _read_tail_lines(path, max_bytes=_MAX_TAIL_BYTES, max_lines=_MAX_TAIL_LINES)
            all_candidate_lines.extend(chunk)

        if all_candidate_lines:
            all_candidate_lines.sort(reverse=True)
            entries = _normalize_log_entries(all_candidate_lines)
            if live_status is None:
                live_status = {}
            live_status["logs"] = entries[:_MAX_LOG_ENTRIES]
    except Exception as e:
        _log_event(
            "warning",
            f"Failed to read logs: {e}",
            key_parameters={"path": str(_LOGS_DIR)},
            stop_reason="log_read_failed",
        )
    return live_status


def _inject_telemetry(
    handler: "PeakStateDashboardHandler",
    live_status: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attaches hardware telemetry and history to live_status."""
    hardware_manager = handler.hardware_manager
    if hardware_manager is None:
        hardware_manager = HardwareManager()
        handler.__class__.hardware_manager = hardware_manager
    telemetry = _get_cached_telemetry(hardware_manager)
    history = _append_hardware_history(telemetry)

    if live_status is None:
        return {"hardware": telemetry, "hardware_history": history}

    live_status["hardware"] = telemetry
    live_status["hardware_history"] = history
    if telemetry.get("gpus"):
        gpu0 = telemetry["gpus"][0]
        live_status.setdefault(
            "gpu_utilization",
            gpu0.get("utilization_total", gpu0.get("utilization")),
        )
        live_status.setdefault("vram_utilization", telemetry["gpus"][0]["vram_usage_pct"])
    live_status.setdefault("ram_utilization", telemetry["ram_usage_pct"])
    return live_status


def _has_live_status_value(value: Any) -> bool:
    """Return whether a live-status field carries meaningful data."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _merge_live_status_sources(
    primary: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge live-status sources while preserving embedded snapshot fields as fallback."""
    if not isinstance(primary, dict):
        return dict(fallback) if isinstance(fallback, dict) else primary
    if not isinstance(fallback, dict):
        return primary

    merged = dict(fallback)
    for key, value in primary.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
            continue
        if _has_live_status_value(value) or key not in merged:
            merged[key] = value
    return merged


def _apply_study_defaults(raw_data: dict[str, Any]) -> None:
    """Sets study-level defaults on raw_data from HPO configuration."""
    study_name = raw_data.get("studyName")
    if not study_name or study_name == "Initializing...":
        study_name = settings.HPO_CONFIG.get("study_name", "PFF HPO Study")
    hpo_defaults = settings.HPO_CONFIG.get("defaults", {})
    raw_data.setdefault("studyName", study_name)
    raw_data.setdefault("direction", hpo_defaults.get("direction", "maximize"))
    raw_data["direction"] = _normalize_direction_label(raw_data.get("direction"))
    raw_data.setdefault("totalTrials", hpo_defaults.get("n_trials", 50))
    raw_data.setdefault("totalFolds", hpo_defaults.get("cv_folds", 1))
    try:
        raw_data["totalFolds"] = max(1, int(raw_data.get("totalFolds", 1)))
    except (TypeError, ValueError):
        raw_data["totalFolds"] = max(1, int(hpo_defaults.get("cv_folds", 1) or 1))
    raw_data.setdefault("charts", {})
    raw_data.setdefault("updatedAt", datetime.now(timezone.utc).isoformat())
    raw_data.setdefault("_synthetic_trials", False)
    raw_data.setdefault(
        "objectiveDirections",
        _infer_objective_directions(raw_data.get("trials"), raw_data.get("direction", "maximize")),
    )
    search_space = raw_data.get("searchSpace")
    if not isinstance(search_space, dict) or not search_space:
        inferred_space = _infer_search_space_from_trials(raw_data.get("trials", []))
        if inferred_space:
            raw_data["searchSpace"] = inferred_space
        else:
            raw_data.setdefault("searchSpace", {})
    _refresh_trial_count_fields(raw_data)


def _sanitize_live_status_fold(
    raw_data: dict[str, Any],
    live_status: dict[str, Any] | None,
) -> None:
    """Normalizes live-status fold id to the configured CV fold range."""
    if not isinstance(live_status, dict):
        return
    try:
        total_folds = max(1, int(raw_data.get("totalFolds", 1)))
    except (TypeError, ValueError):
        total_folds = 1

    raw_fold_id = live_status.get("cv_fold_id")
    if raw_fold_id is None:
        return
    try:
        fold_id = int(raw_fold_id)
    except (TypeError, ValueError):
        live_status["cv_fold_id"] = 0
        _log_event(
            "warning",
            "Dashboard received non-integer cv_fold_id and reset it to fold 0.",
            key_parameters={"cv_fold_id": raw_fold_id, "total_folds": total_folds},
            stop_reason="live_status_cv_fold_invalid_type",
        )
        return

    if fold_id < 0 or fold_id >= total_folds:
        live_status["cv_fold_id"] = 0
        _log_event(
            "warning",
            "Dashboard received out-of-range cv_fold_id and reset it to fold 0.",
            key_parameters={"cv_fold_id": fold_id, "total_folds": total_folds},
            stop_reason="live_status_cv_fold_out_of_range",
        )
        return
    live_status["cv_fold_id"] = fold_id


def _infer_search_space_from_trials(trials: Any) -> dict[str, Any]:
    """Infer minimal search-space metadata from trial params when distributions are unavailable."""
    if not isinstance(trials, list):
        return {}

    values_by_param: dict[str, list[Any]] = {}
    for trial in trials:
        if not isinstance(trial, Mapping):
            continue
        params = trial.get("params")
        if not isinstance(params, Mapping):
            continue
        for key, value in params.items():
            if value is None:
                continue
            name = str(key)
            values_by_param.setdefault(name, []).append(value)

    inferred: dict[str, Any] = {}
    for name, values in values_by_param.items():
        if not values:
            continue

        if all(isinstance(v, bool) for v in values):
            choices = sorted({bool(v) for v in values})
            inferred[name] = {"type": "categorical", "choices": choices}
            continue

        numeric_values = [v for v in values if type(v) is not bool and isinstance(v, Real)]
        if len(numeric_values) == len(values):
            all_int = all(type(v) is not bool and isinstance(v, Integral) for v in values)
            if all_int:
                sorted_unique = sorted({int(cast(Integral, v)) for v in numeric_values})
            else:
                sorted_unique = sorted({float(v) for v in numeric_values})
            low_value = sorted_unique[0]
            high_value = sorted_unique[-1]

            if low_value == high_value:
                inferred[name] = {"type": "fixed", "value": low_value}
                continue

            if all_int and len(sorted_unique) <= 12:
                inferred[name] = {"type": "categorical", "choices": sorted_unique}
                continue

            entry: dict[str, Any] = {
                "type": "int" if all_int else "float",
                "low": low_value,
                "high": high_value,
            }
            if (
                not all_int
                and isinstance(low_value, float)
                and isinstance(high_value, float)
                and low_value > 0.0
                and (high_value / low_value) >= 100.0
            ):
                entry["log"] = True
            inferred[name] = entry
            continue

        unique_values: list[Any] = []
        seen_keys: set[str] = set()
        for value in values:
            key = repr(value)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_values.append(
                value if isinstance(value, (str, int, float, bool)) else str(value)
            )

        if len(unique_values) == 1:
            inferred[name] = {"type": "fixed", "value": unique_values[0]}
        else:
            inferred[name] = {"type": "categorical", "choices": unique_values[:32]}

    return inferred


def _infer_objective_directions(trials: Any, fallback_direction: str) -> list[str]:
    """Infer objective directions from trial vectors when available."""
    norm_fallback = _normalize_direction_label(fallback_direction)
    if not isinstance(trials, list):
        return [norm_fallback]
    max_len = 0
    for trial in trials:
        if not isinstance(trial, Mapping):
            continue
        values = trial.get("values")
        if isinstance(values, (list, tuple)):
            max_len = max(max_len, len(values))
    if max_len <= 0:
        return [norm_fallback]
    return [norm_fallback for _ in range(max_len)]


def _apply_debug_mode(
    raw_data: dict[str, Any],
    live_status: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Applies dashboard debug mode overrides and returns updated live_status."""
    live_plot_settings = load_live_plot_settings()
    debug_mode = bool(live_plot_settings.get("dashboard_debug_mode", False))
    raw_data["dashboardDebugMode"] = debug_mode

    if not debug_mode:
        return live_status

    if not isinstance(live_status, dict):
        live_status = {}
    debug_status: dict[str, Any] = {
        **live_status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
    }
    if isinstance(raw_data.get("trials"), list):
        valid_ids = [t.get("id") for t in raw_data["trials"] if isinstance(t, dict)]
        valid_ids = [int(tid) for tid in valid_ids if isinstance(tid, int)]
        if valid_ids:
            debug_status["trial_number"] = max(v for v in valid_ids if v is not None) - 1
    raw_data["updatedAt"] = debug_status["updated_at"]
    return debug_status


def _consolidate_live_trial(
    raw_data: dict[str, Any],
    live_status: dict[str, Any] | None,
) -> None:
    """Merges current running trial data from live_status into raw_data trials."""
    identifiers = _extract_live_identifiers(live_status)
    if identifiers is None:
        return
    live_id, synthetic_id = identifiers

    debug_mode = raw_data.get("dashboardDebugMode", False)
    valid_files_exist = any(p.exists() for p in _collect_dashboard_data_paths())

    try:
        trials_list = raw_data.get("trials", [])
        trials_map = _normalize_trials_map(trials_list, synthetic_id)

        epoch_history = (
            live_status.get("epoch_history", []) if isinstance(live_status, dict) else []
        )
        latest_epoch_metrics = _extract_latest_epoch(epoch_history)
        best_epoch_metrics, best_epoch_score = _extract_best_epoch(epoch_history)

        _sync_live_status_trial_number(raw_data, live_id)

        live_row = _find_live_row(trials_map, live_id)
        if live_row is None and valid_files_exist and not debug_mode:
            live_row = _create_live_trial_row(trials_map, live_id, live_status)

        if live_row:
            best_epoch_metrics = _update_lookback_best(live_id, live_row, best_epoch_metrics)

        if live_row and best_epoch_metrics and live_row.get("state") != "COMPLETE":
            _merge_epoch_into_trial(
                live_row,
                best_epoch_metrics,
                best_epoch_score,
                live_status,
                latest_epoch_metrics,
            )

        running_trial_id = int(live_id) + 1
        if not valid_files_exist and running_trial_id not in trials_map and not debug_mode:
            _create_synthetic_trial(trials_map, synthetic_id, live_status, raw_data)

        raw_data["trials"] = sorted(trials_map.values(), key=lambda x: int(x.get("id", 0)))

    except (ValueError, TypeError) as e:
        _log_event(
            "warning",
            f"Failed to consolidate live data: {e}",
            key_parameters={"study": raw_data.get("studyName")},
            stop_reason="live_data_consolidation_failed",
        )


def _extract_live_identifiers(
    live_status: dict[str, Any] | None,
) -> tuple[float, int] | None:
    """Execute extract live identifiers.



    Args:

        live_status: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if not isinstance(live_status, dict) or "trial_number" not in live_status:
        return None
    trial_val = live_status.get("trial_number")
    if trial_val is None or not isinstance(trial_val, (int, float)):
        return None
    live_id = float(trial_val)
    synthetic_id = -abs(int(live_id)) if int(live_id) != 0 else -1
    return live_id, synthetic_id


def _normalize_trials_map(
    trials_list: Any,
    synthetic_id: int,
) -> dict[Any, dict[str, Any]]:
    """Execute normalize trials map.



    Args:

        trials_list: Input value used by this callable.

        synthetic_id: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if not isinstance(trials_list, list):
        return {}

    trials_map: dict[Any, dict[str, Any]] = {}
    for trial in trials_list:
        if not isinstance(trial, dict):
            continue
        trial_id = trial.get("id")
        if trial_id is None:
            continue
        if isinstance(trial_id, (int, float)):
            trial_id = abs(int(trial_id))
            trial["id"] = trial_id
        if trial_id == synthetic_id:
            continue
        trials_map[trial_id] = trial
    return trials_map


def _sync_live_status_trial_number(raw_data: dict[str, Any], live_id: float) -> None:
    """Execute sync live status trial number.



    Args:

        raw_data: Input value used by this callable.

        live_id: Input value used by this callable.

    """

    if isinstance(raw_data.get("liveStatus"), dict):
        raw_data["liveStatus"]["trial_number"] = live_id


def _extract_best_epoch(
    epoch_history: list,
) -> tuple[dict[str, Any] | None, float]:
    """Finds the best epoch metrics from the epoch history."""
    if not epoch_history:
        return None, 0.0
    metrics_list = [e for e in epoch_history if isinstance(e, dict)]
    if not metrics_list:
        return None, 0.0
    scored_epochs = [e for e in metrics_list if _has_metrics(e)]
    best = max(scored_epochs, key=_epoch_score) if scored_epochs else metrics_list[-1]
    return best, _epoch_score(best)


def _extract_latest_epoch(epoch_history: list) -> dict[str, Any] | None:
    """Returns the latest epoch metrics dict from epoch_history."""
    if not epoch_history:
        return None
    metrics_list = [e for e in epoch_history if isinstance(e, dict)]
    if not metrics_list:
        return None
    return metrics_list[-1]


def _find_live_row(trials_map: dict, live_id: float) -> dict[str, Any] | None:
    """Finds the trial row matching the current live trial."""
    try:
        base_id = int(live_id)
    except (TypeError, ValueError):
        return None

    preferred = trials_map.get(base_id + 1)
    if isinstance(preferred, dict):
        if preferred.get("state") != "COMPLETE":
            return preferred

    legacy = trials_map.get(base_id)
    if isinstance(legacy, dict) and legacy.get("state") != "COMPLETE":
        return legacy

    return preferred if isinstance(preferred, dict) else None


def _create_live_trial_row(
    trials_map: dict[Any, dict[str, Any]],
    live_id: float,
    live_status: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Creates a RUNNING trial row from live_status when the live trial is absent."""
    live_status_payload = live_status or {}
    try:
        trial_id = int(live_id) + 1
    except (TypeError, ValueError):
        return None

    existing = trials_map.get(trial_id)
    if isinstance(existing, dict):
        return existing

    epoch_history = live_status_payload.get("epoch_history", [])
    latest = _extract_latest_epoch(epoch_history) if isinstance(epoch_history, list) else None
    best, best_score = (
        _extract_best_epoch(epoch_history) if isinstance(epoch_history, list) else (None, 0.0)
    )
    duration = float(live_status_payload.get("elapsed_seconds", 0.0) or 0.0)
    loss_value = _resolve_loss_value(best, latest)
    metrics_payload = _clean_metrics(latest or {})
    if duration:
        metrics_payload.setdefault("duration", duration)
    if loss_value is not None:
        metrics_payload.setdefault("loss", loss_value)

    row: dict[str, Any] = {
        "id": trial_id,
        "value": best_score,
        "state": "RUNNING",
        "params": live_status_payload.get("params", {}),
        "duration": duration,
        "loss": loss_value,
        "warmstart": bool(live_status_payload.get("warmstart")),
        "metrics": metrics_payload,
    }
    trials_map[trial_id] = row
    return row


def _update_lookback_best(
    live_id: float,
    live_row: dict[str, Any],
    best_epoch_metrics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Updates lookback memory with best metrics for the live trial."""
    try:
        live_key = str(int(live_id))
    except (TypeError, ValueError):
        return best_epoch_metrics

    with _LOOKBACK_LOCK:
        lookback = _get_lookback_memory()
        live_best = lookback.get("live_best_metrics")
        if not isinstance(live_best, dict):
            live_best = {}
            lookback["live_best_metrics"] = live_best

        previous_best = live_best.get(live_key, {})
        if best_epoch_metrics:
            cleaned = {k: v for k, v in best_epoch_metrics.items() if v is not None}
            merged = {**previous_best, **cleaned}
            live_best[live_key] = merged
            best_epoch_metrics = merged
        elif previous_best:
            best_epoch_metrics = previous_best
        _set_lookback_memory(lookback)

    if live_row.get("state") == "COMPLETE":
        with _LOOKBACK_LOCK:
            lookback = _get_lookback_memory()
            live_best = lookback.get("live_best_metrics")
            if isinstance(live_best, dict):
                live_best.pop(live_key, None)
                lookback["live_best_metrics"] = live_best
                _set_lookback_memory(lookback)

    return best_epoch_metrics


def _merge_epoch_into_trial(
    live_row: dict[str, Any],
    best_epoch_metrics: dict[str, Any],
    best_epoch_score: float,
    live_status: dict[str, Any] | None,
    latest_epoch_metrics: dict[str, Any] | None,
) -> None:
    """Merges best epoch metrics into the live trial row."""
    loss_value = _resolve_loss_value(best_epoch_metrics, latest_epoch_metrics)
    duration, efficiency = _compute_duration_and_efficiency(live_status, best_epoch_score)
    metrics_payload = _build_metrics_payload(
        live_row, best_epoch_metrics, loss_value, duration, efficiency
    )
    update_payload = _build_trial_update_payload(
        metrics_payload=metrics_payload,
        best_epoch_metrics=best_epoch_metrics,
        loss_value=loss_value,
        duration=duration,
        efficiency=efficiency,
    )
    live_row.update(update_payload)


def _resolve_loss_value(
    best_epoch_metrics: dict[str, Any] | None,
    latest_epoch_metrics: dict[str, Any] | None = None,
) -> Any:
    for metrics in (latest_epoch_metrics, best_epoch_metrics):
        if not isinstance(metrics, dict):
            continue
        loss_value = (
            metrics.get("loss")
            or metrics.get("val_loss")
            or metrics.get("train_loss")
            or metrics.get("binary_loss")
        )
        if loss_value is not None:
            return loss_value
    return None


def _compute_duration_and_efficiency(
    live_status: dict[str, Any] | None,
    best_epoch_score: float,
) -> tuple[float, float | None]:
    """Execute compute duration and efficiency.



    Args:

        live_status: Input value used by this callable.

        best_epoch_score: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    live_status_payload = live_status or {}
    duration = float(live_status_payload.get("elapsed_seconds", 0.0) or 0.0)
    if not duration:
        return 0.0, None
    try:
        return duration, float(best_epoch_score) / duration
    except (TypeError, ValueError):
        return duration, None


def _build_metrics_payload(
    live_row: dict[str, Any],
    best_epoch_metrics: dict[str, Any],
    loss_value: Any,
    duration: float,
    efficiency: float | None,
) -> dict[str, Any]:
    """Execute build metrics payload.



    Args:

        live_row: Input value used by this callable.

        best_epoch_metrics: Input value used by this callable.

        loss_value: Input value used by this callable.

        duration: Input value used by this callable.

        efficiency: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    raw_metrics = live_row.get("metrics")
    metrics_payload = raw_metrics if isinstance(raw_metrics, dict) else {}
    metrics_payload = {**metrics_payload, **_clean_metrics(best_epoch_metrics)}
    if duration:
        metrics_payload.setdefault("duration", duration)
    if efficiency is not None:
        metrics_payload.setdefault("efficiency", efficiency)
    if loss_value is not None:
        metrics_payload.setdefault("loss", loss_value)
    return metrics_payload


def _build_trial_update_payload(
    *,
    metrics_payload: dict[str, Any],
    best_epoch_metrics: dict[str, Any],
    loss_value: Any,
    duration: float,
    efficiency: float | None,
) -> dict[str, Any]:
    """Execute build trial update payload.



    Args:

        metrics_payload: Input value used by this callable.

        best_epoch_metrics: Input value used by this callable.

        loss_value: Input value used by this callable.

        duration: Input value used by this callable.

        efficiency: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    update_payload: dict[str, Any] = {"metrics": metrics_payload}
    if loss_value is not None:
        update_payload["loss"] = loss_value
    if efficiency is not None:
        update_payload["efficiency"] = efficiency
    if duration:
        update_payload["duration"] = duration

    for key in ("precision", "recall", "mrr", "mcc", "accuracy", "f1", "auc"):
        value = best_epoch_metrics.get(key)
        if value is not None:
            update_payload[key] = value

    for original_key, normalized_key in (
        ("hits@1", "hits1"),
        ("hits@3", "hits3"),
        ("hits@10", "hits10"),
    ):
        value = best_epoch_metrics.get(original_key)
        if value is None:
            value = best_epoch_metrics.get(normalized_key)
        if value is not None:
            update_payload[normalized_key] = value
    return update_payload


def _create_synthetic_trial(
    trials_map: dict,
    synthetic_id: int,
    live_status: dict[str, Any] | None,
    raw_data: dict[str, Any],
) -> None:
    """Creates a synthetic RUNNING trial entry when no dashboard files exist."""
    live_status_payload = live_status or {}
    epoch_history = live_status_payload.get("epoch_history", [])
    live_score = 0.0
    last_metrics: dict[str, Any] = {}
    best_metrics = {"mrr": 0.0, "mcc": 0.0}

    if isinstance(epoch_history, list) and epoch_history:
        metrics_list = [e for e in epoch_history if isinstance(e, dict)]
        if metrics_list:
            live_score = max(float(e.get("mrr") or e.get("score") or 0.0) for e in metrics_list)
            last_metrics = metrics_list[-1]
            best_metrics["mrr"] = max(float(e.get("mrr", 0.0)) for e in metrics_list)
            best_metrics["mcc"] = max(float(e.get("mcc", 0.0)) for e in metrics_list)

    loss_value = (
        last_metrics.get("loss")
        or last_metrics.get("val_loss")
        or last_metrics.get("train_loss")
        or last_metrics.get("binary_loss")
    )
    duration = float(live_status_payload.get("elapsed_seconds", 0.0) or 0.0)
    efficiency = None
    if duration:
        try:
            efficiency = float(live_score) / duration
            last_metrics.setdefault("efficiency", efficiency)
        except (TypeError, ValueError):
            efficiency = None
    last_metrics.setdefault("duration", duration)
    if loss_value is not None:
        last_metrics.setdefault("loss", loss_value)

    warmstart_flag = bool(
        live_status_payload.get("warmstart") or live_status_payload.get("warmstart_seed")
    )

    trials_map[synthetic_id] = {
        "id": synthetic_id,
        "value": live_score,
        "state": "RUNNING",
        "params": live_status_payload.get("params", {}),
        "duration": duration,
        "loss": loss_value,
        "precision": last_metrics.get("precision"),
        "recall": last_metrics.get("recall"),
        "efficiency": efficiency,
        "warmstart": warmstart_flag,
        "mrr": last_metrics.get("mrr"),
        "best_mrr": best_metrics["mrr"],
        "mcc": last_metrics.get("mcc"),
        "best_mcc": best_metrics["mcc"],
        "accuracy": last_metrics.get("accuracy"),
        "f1": last_metrics.get("f1"),
        "auc": last_metrics.get("auc"),
        "hits1": last_metrics.get("hits@1"),
        "hits3": last_metrics.get("hits@3"),
        "hits10": last_metrics.get("hits@10"),
        "metrics": last_metrics,
    }
    raw_data["_synthetic_trials"] = True


def _compute_best_value(raw_data: dict[str, Any]) -> None:
    """Computes the best trial value across all trials."""
    all_trials = raw_data.get("trials", [])
    if not isinstance(all_trials, list) or not all_trials:
        return
    valid_values = [
        float(t.get("value", 0.0))
        for t in all_trials
        if isinstance(t, dict) and t.get("value") is not None
    ]
    if valid_values:
        raw_data["bestValue"] = max(valid_values)


def _update_fold_memory(
    lookback: dict[str, Any],
    confusion_matrix: dict[str, Any] | None,
    current: dict[str, Any],
) -> None:
    """Updates fold confusion matrix history in lookback memory."""
    if not isinstance(confusion_matrix, dict):
        return
    trial_number = current.get("trial_number")
    cv_fold_id = current.get("cv_fold_id")
    if trial_number is None or cv_fold_id is None:
        return

    entry = {
        "trial_number": trial_number,
        "cv_fold_id": cv_fold_id,
        "epoch": current.get("current_epoch"),
        "confusion_matrix": confusion_matrix,
    }
    existing = lookback.get("confusion_matrices")
    history = existing if isinstance(existing, list) else []
    if (
        history
        and history[-1].get("cv_fold_id") == cv_fold_id
        and history[-1].get("trial_number") == trial_number
    ):
        history[-1] = entry
    else:
        history = history + [entry]
    lookback["confusion_matrices"] = history[-3:]


def _apply_lookback_memory(
    raw_data: dict[str, Any],
    live_status: dict[str, Any] | None,
) -> None:
    """Applies lookback memory for validation charts when fresh data is unavailable."""
    charts = raw_data.get("charts", {})
    if not isinstance(charts, dict):
        charts = {}
        raw_data["charts"] = charts

    current = live_status or {}

    has_fresh_validation = bool(
        charts.get("gen_gap") or charts.get("confusion_matrix") or charts.get("confusion_matrices")
    )

    with _LOOKBACK_LOCK:
        lookback = _get_lookback_memory()
        if has_fresh_validation:
            lookback["gen_gap"] = charts.get("gen_gap")
            lookback["confusion_matrix"] = charts.get("confusion_matrix")
            _update_fold_memory(lookback, charts.get("confusion_matrix"), current)
            if charts.get("confusion_matrices"):
                lookback["confusion_matrices"] = charts.get("confusion_matrices")
            lookback["last_valid_epoch"] = current.get("current_epoch", -1)
            lookback["source_trial"] = current.get("trial_number", -1)
            raw_data["stale_validation"] = False
        else:
            raw_data["stale_validation"] = True
            if lookback["gen_gap"]:
                charts["gen_gap"] = lookback["gen_gap"]
            if lookback["confusion_matrix"]:
                charts["confusion_matrix"] = lookback["confusion_matrix"]
            if lookback["confusion_matrices"]:
                charts["confusion_matrices"] = lookback["confusion_matrices"]
            charts["lookback_epoch"] = lookback["last_valid_epoch"]
            charts["lookback_trial"] = lookback["source_trial"]
        _set_lookback_memory(lookback)


def _augment_confusion_matrices_from_fold_history(
    raw_data: dict[str, Any],
    live_status: dict[str, Any] | None,
) -> None:
    """Populate charts.confusion_matrices from fold_history.json + live fold when available."""
    charts = raw_data.get("charts")
    if not isinstance(charts, dict):
        charts = {}
        raw_data["charts"] = charts

    merged: dict[tuple[Any, Any], dict[str, Any]] = {}

    existing = charts.get("confusion_matrices")
    if isinstance(existing, list):
        for row in existing:
            if not isinstance(row, dict):
                continue
            cm = row.get("confusion_matrix")
            if not isinstance(cm, dict):
                continue
            key = (row.get("trial_number"), row.get("cv_fold_id"))
            merged[key] = {
                "trial_number": row.get("trial_number"),
                "cv_fold_id": row.get("cv_fold_id"),
                "epoch": row.get("epoch"),
                "timestamp": row.get("timestamp"),
                "confusion_matrix": cm,
            }

    fold_history_candidates = [
        BASE_DIR / "outputs" / "optimization" / "plots" / "fold_history.json",
        settings.OUTPUTS_DIR / "optimization" / "plots" / "fold_history.json",
    ]
    for fold_history_path in fold_history_candidates:
        if not fold_history_path.exists():
            continue
        try:
            history_data = FileManager().read(fold_history_path)
            if hasattr(history_data, "to_native"):
                history_data = history_data.to_native()
        except Exception:
            history_data = []
        if not isinstance(history_data, list):
            continue
        for row in history_data:
            if not isinstance(row, dict):
                continue
            cm = row.get("confusion_matrix")
            if not isinstance(cm, dict):
                continue
            key = (row.get("trial_number"), row.get("cv_fold_id"))
            merged[key] = {
                "trial_number": row.get("trial_number"),
                "cv_fold_id": row.get("cv_fold_id"),
                "epoch": row.get("epoch"),
                "timestamp": row.get("timestamp"),
                "confusion_matrix": cm,
            }
        break

    if isinstance(live_status, dict) and isinstance(live_status.get("confusion_matrix"), dict):
        key = (live_status.get("trial_number"), live_status.get("cv_fold_id"))
        merged[key] = {
            "trial_number": live_status.get("trial_number"),
            "cv_fold_id": live_status.get("cv_fold_id"),
            "epoch": live_status.get("current_epoch"),
            "timestamp": time.time(),
            "confusion_matrix": live_status.get("confusion_matrix"),
        }
        charts.setdefault("confusion_matrix", live_status.get("confusion_matrix"))

    if not merged:
        return

    items = list(merged.values())
    items.sort(key=lambda row: (float(row.get("timestamp") or 0.0), int(row.get("epoch") or 0)))
    charts["confusion_matrices"] = items[-3:]


class PeakStateDashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for Peak State performance and robustness."""

    protocol_version = "HTTP/1.1"
    hardware_manager: HardwareManager | None = None
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
    }

    def __init__(self, *args, **kwargs):
        """Execute init.



        Args:

            *args: Additional positional arguments.

            **kwargs: Additional keyword arguments.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if self.__class__.hardware_manager is None:
            self.__class__.hardware_manager = HardwareManager()
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def translate_path(self, path: str) -> str:
        """Resolve /static and /dist assets against the patched dashboard roots."""
        clean_path = path.split("?", 1)[0].split("#", 1)[0]
        for prefix, root_dir in (("/static/", STATIC_DIR), ("/dist/", DIST_DIR)):
            if clean_path == prefix[:-1] or clean_path.startswith(prefix):
                relative = clean_path[len(prefix) :] if clean_path.startswith(prefix) else ""
                safe_parts = [
                    part for part in Path(unquote(relative)).parts if part not in {"", ".", ".."}
                ]
                return str(root_dir.joinpath(*safe_parts))
        return super().translate_path(path)

    def do_GET(self):
        """Routes API requests or serves static files (Strict Dispatcher - Final)."""
        clean_path = self.path.split("?")[0]

        self._force_connection_close = True

        if clean_path == "/api/events" or clean_path == "/api/events/":
            self._force_connection_close = False
            try:
                self._serve_sse()
            except Exception as e:
                _log_event(
                    "error",
                    f"SSE handler failed: {e}",
                    key_parameters={"path": clean_path},
                    stop_reason="sse_handler_failed",
                )
                self.send_error(500, str(e))
            return

        if clean_path.startswith("/api/"):
            self._serve_dashboard_api()
            return

        if clean_path.startswith(("/dist/", "/static/")):
            super().do_GET()
            return

        if clean_path == "/" or clean_path == "/index.html":
            self._serve_index_with_build_id()
            return

        translated = Path(self.translate_path(clean_path))
        if FileManager.exists(translated) and translated.is_file():
            super().do_GET()
            return

        if "api" in clean_path:
            _log_event(
                "warning",
                f"API path fell through to SPA fallback: {clean_path}",
                key_parameters={"path": clean_path},
                stop_reason="api_fallback",
            )

        self.path = "/static/index.html"
        self._serve_index_with_build_id()

    def _serve_index_with_build_id(self):
        """Serve index.html with a build id query param for /dist assets.

        This avoids stale bundles when /dist is cached as immutable.
        """
        self.path = "/static/index.html"

        index_path = STATIC_DIR / "index.html"

        build_id_path = DIST_DIR / "build_id.txt"
        build_id = None
        try:
            if FileManager.exists(build_id_path):
                build_id = (
                    FileManager.read_bytes(build_id_path).decode("utf-8", errors="ignore").strip()
                )
        except Exception:
            build_id = None

        if not build_id:
            build_id = str(int(time.time()))

        try:
            raw = FileManager.read_bytes(index_path).decode("utf-8", errors="strict")
        except Exception as e:
            _log_event(
                "error",
                f"Failed to read index.html: {e}",
                key_parameters={"path": str(index_path)},
                stop_reason="index_read_failed",
            )
            self.send_error(500, "Failed to load index")
            return

        html = raw.replace("__ts__", build_id)
        html = html.replace(
            "</body>",
            f'<script>window.__PFF_BUILD_ID__ = "{build_id}";</script></body>',
        )
        content = html.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        try:
            self.wfile.write(content)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self):
        """Handle export requests via strategy dispatch."""
        self._force_connection_close = True
        if self.path.startswith("/api/export"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = _load_json_payload(body)
                fmt = payload.get("format", "json")
                filename = payload.get("filename", "export")
                data = payload.get("data", {})

                handler = _EXPORT_HANDLERS.get(fmt, _export_json)
                content, content_type = handler(data)

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Disposition", f'attachment; filename="{filename}.{fmt}"')
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                try:
                    self.wfile.write(content)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                return

            except Exception as e:
                _log_event(
                    "error",
                    f"Export failed: {e}",
                    key_parameters={"path": self.path},
                    stop_reason="export_failed",
                )
                self.send_error(500, f"Export failed: {str(e)}")
                return

        if self.path.startswith("/api/hpo/search-space-advice/patch"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = _load_json_payload(body)
                recommendations = payload.get("recommendations", [])
                patch = generate_search_space_patch(recommendations)
                data = {"patch": patch, "n_changes": len(patch)}
                content = FileManager.json_dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                try:
                    self.wfile.write(content)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                return
            except Exception as e:
                self.send_error(500, f"Patch generation failed: {str(e)}")
                return

        if self.path.startswith("/api/hpo/search-space-advice/apply"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = _load_json_payload(body)
                recommendations = payload.get("recommendations", [])
                if not isinstance(recommendations, list) or not recommendations:
                    self.send_error(400, "No recommendations provided")
                    return
                patch = generate_search_space_patch(recommendations)
                clear_config_cache()
                config = _load_optimization_config_rt()
                config, applied, skipped = _apply_search_space_patch_to_config(config, patch)
                if applied:
                    YAMLHandler().save(config, Path(OPTIMIZATION_CONFIG_PATH))
                    clear_config_cache()
                updated_paths = _update_dashboard_payloads(patch, applied_params=applied)
                _log_event(
                    "info",
                    "Aplicadas recomendacoes do advisor ao YAML de otimizacao.",
                    key_parameters={
                        "applied": applied,
                        "skipped": skipped,
                        "paths": updated_paths,
                    },
                    stop_reason="search_space_advice_applied",
                )
                data = {
                    "status": "ok",
                    "applied_params": applied,
                    "skipped_params": skipped,
                    "updated_paths": updated_paths,
                }
                content = FileManager.json_dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                try:
                    self.wfile.write(content)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                return
            except Exception as e:
                self.send_error(500, f"Apply failed: {str(e)}")
                return

        if self.path.startswith("/api/hpo/search-space-advice/ignore"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = _load_json_payload(body)
                param_names = payload.get("param_names", [])
                if isinstance(param_names, str):
                    param_names = [param_names]
                if not isinstance(param_names, list) or not param_names:
                    self.send_error(400, "No param_names provided")
                    return
                param_names = [str(name) for name in param_names]
                updated_paths = _update_dashboard_payloads({}, ignored_params=param_names)
                _log_event(
                    "info",
                    "Ignorados parametros do advisor no dashboard.",
                    key_parameters={"ignored": param_names, "paths": updated_paths},
                    stop_reason="search_space_advice_ignored",
                )
                data = {
                    "status": "ok",
                    "ignored_params": param_names,
                    "updated_paths": updated_paths,
                }
                content = FileManager.json_dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                try:
                    self.wfile.write(content)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                return
            except Exception as e:
                self.send_error(500, f"Ignore failed: {str(e)}")
                return

        self.send_error(501, "Unsupported method ('POST')")

    def do_OPTIONS(self):
        """Handle CORS preflight for API routes."""
        self._force_connection_close = True
        clean_path = self.path.split("?")[0]
        if not clean_path.startswith("/api/"):
            self.send_error(404)
            return

        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def end_headers(self):
        """Inject strict Peak State caching policies."""
        file_path = self.path.split("?")[0]

        if file_path.endswith(".html") or file_path == "/":
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")

        elif file_path.startswith("/dist/") and (
            file_path.endswith(".js") or file_path.endswith(".css") or file_path.endswith(".wasm")
        ):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")

        elif file_path.endswith((".js", ".css", ".woff2", ".png", ".svg")):
            self.send_header("Cache-Control", "public, max-age=3600")

        if file_path.startswith("/api/"):
            self.send_header("Access-Control-Allow-Origin", "*")

        if getattr(self, "_force_connection_close", False):
            self.send_header("Connection", "close")
            self.close_connection = True
        super().end_headers()

    def _serve_sse(self):
        """Serves Server-Sent Events for real-time dashboard updates."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        sse_event_id = 0

        def write_sse_payload(payload: str) -> None:
            nonlocal sse_event_id
            sse_event_id += 1
            self.wfile.write(f"id: {sse_event_id}\n".encode("utf-8"))
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))

        self.wfile.write(b": ping\n\n")
        self.wfile.write(b"retry: 3000\n\n")
        self.wfile.flush()

        try:
            data = self._load_consolidated_data()
            payload = FileManager.json_dumps(data)
            write_sse_payload(payload)
            self.wfile.flush()
        except Exception as e:
            _log_event(
                "error",
                f"SSE initial push failed: {e}",
                key_parameters={"client": str(self.client_address)},
                stop_reason="sse_initial_push_failed",
            )

        last_mtime = 0.0
        last_live_mtime = 0.0
        last_push = 0.0

        _log_event(
            "debug",
            f"SSE client connected: {self.client_address}",
            key_parameters={"client": str(self.client_address)},
            stop_reason="sse_client_connected",
        )

        try:
            while not should_stop():
                paths = _collect_dashboard_data_paths()
                valid_files = [p for p in paths if FileManager.exists(p)]
                current_mtime = (
                    max([p.stat().st_mtime for p in valid_files]) if valid_files else 0.0
                )

                live_status_path = (
                    BASE_DIR / "outputs" / "optimization" / "plots" / "live_status.json"
                )
                current_live_mtime = (
                    live_status_path.stat().st_mtime
                    if FileManager.exists(live_status_path)
                    else 0.0
                )
                now = time.time()
                if (
                    current_mtime > last_mtime
                    or current_live_mtime > last_live_mtime
                    or now - last_push >= 1.0
                ):
                    data = self._load_consolidated_data()
                    payload = FileManager.json_dumps(data)
                    write_sse_payload(payload)
                    self.wfile.flush()
                    last_mtime = current_mtime
                    last_live_mtime = current_live_mtime
                    last_push = now

                time.sleep(1)
        except (ConnectionResetError, BrokenPipeError):
            _log_event(
                "debug",
                f"SSE client disconnected: {self.client_address}",
                key_parameters={"client": str(self.client_address)},
                stop_reason="sse_client_disconnected",
            )
        except Exception as e:
            _log_event(
                "error",
                f"SSE error: {e}",
                key_parameters={"client": str(self.client_address)},
                stop_reason="sse_error",
            )

    def _serve_dashboard_api(self):
        """Consolidates HPO data with Lookback Logic."""
        clean_path = self.path.split("?")[0]
        if clean_path.startswith("/api/status"):
            data = {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
        elif clean_path == "/api/hpo/search-space-advice":
            data = self._serve_search_space_advice()
        else:
            data = self._load_consolidated_data()
        content = FileManager.json_dumps(data).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        try:
            self.wfile.write(content)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _serve_search_space_advice(self) -> dict[str, Any]:
        raw_data = self._load_consolidated_data()
        force_refresh = _query_flag(self.path, "refresh") or _query_flag(self.path, "recompute")
        cached_advice = raw_data.get("searchSpaceAdvice")
        if (
            not force_refresh
            and isinstance(cached_advice, dict)
            and _has_usable_search_space_advice(cached_advice)
        ):
            return cached_advice
        try:
            advisor = _get_search_space_advisor()
            search_space = raw_data.get("searchSpace", {})
            trials = raw_data.get("trials", [])
            importances = raw_data.get("importances", {})
            direction = raw_data.get("direction", "maximize")
            study_name = raw_data.get("studyName", "")
            objective_directions = raw_data.get("objectiveDirections")
            dataset_fingerprint = None
            dataset_profile = None
            try:
                dataset_fingerprint, dataset_profile = compute_dataset_profile_fingerprint()
            except Exception as e:
                _log_event(
                    "warning",
                    f"Search space dataset profiling failed: {e}",
                    key_parameters={},
                    stop_reason="search_space_dataset_profile_failed",
                )
            return advisor.advise(
                search_space=search_space,
                trials_data=trials,
                importances=importances,
                direction=direction,
                study_name=study_name,
                dataset_fingerprint=dataset_fingerprint,
                dataset_profile=dataset_profile,
                objective_directions=(
                    objective_directions if isinstance(objective_directions, list) else None
                ),
                advisor_config=(
                    raw_data.get("searchSpaceCoverage")
                    if isinstance(raw_data.get("searchSpaceCoverage"), dict)
                    else None
                ),
                force_recompute=force_refresh,
            )
        except Exception as e:
            _log_event(
                "warning",
                f"Search space advice request failed: {e}",
                key_parameters={},
                stop_reason="search_space_advice_request_failed",
            )
            return cached_advice if isinstance(cached_advice, dict) else {}

    def _load_consolidated_data(self) -> dict[str, Any]:
        """Loads data from multi-path and applies lookback memory + live consolidation."""
        initial_live_status = _load_live_status()
        preferred_live_trial_id = (
            _trial_id_from_live_status(initial_live_status)
            if isinstance(initial_live_status, dict)
            else None
        )
        active_study_name = os.getenv("PFF_ACTIVE_STUDY_NAME")
        if not active_study_name and isinstance(initial_live_status, dict):
            status_study_name = initial_live_status.get("study_name")
            if isinstance(status_study_name, str) and status_study_name.strip():
                active_study_name = status_study_name.strip()
        if not active_study_name:
            cfg_study_name = settings.HPO_CONFIG.get("study_name")
            if isinstance(cfg_study_name, str) and cfg_study_name.strip():
                active_study_name = cfg_study_name.strip()
        raw_data = _load_raw_dashboard_data(
            active_study_name=active_study_name,
            preferred_live_trial_id=preferred_live_trial_id,
        )
        raw_live_status = raw_data.get("liveStatus")
        embedded_live_status = (
            dict(cast(Mapping[str, Any], raw_live_status))
            if isinstance(raw_live_status, dict)
            else None
        )
        if not active_study_name:
            raw_study = raw_data.get("studyName")
            if isinstance(raw_study, str) and raw_study.strip():
                active_study_name = raw_study.strip()

        raw_data.setdefault("trials", [])
        preferred_trial_ids = _extract_active_trial_ids(raw_data)
        live_status = _load_live_status(
            preferred_study_name=active_study_name,
            preferred_trial_ids=preferred_trial_ids,
        )
        live_status = _merge_live_status_sources(live_status, initial_live_status)
        live_status = _merge_live_status_sources(live_status, embedded_live_status)

        live_status = _collect_terminal_logs(live_status, raw_data)

        live_status = _inject_telemetry(self, live_status)

        raw_data["liveStatus"] = live_status

        _apply_study_defaults(raw_data)

        live_status = _apply_debug_mode(raw_data, live_status)
        _sanitize_live_status_fold(raw_data, live_status)
        raw_data["liveStatus"] = live_status

        _consolidate_live_trial(raw_data, live_status)
        _refresh_trial_count_fields(raw_data)

        _compute_best_value(raw_data)

        _augment_confusion_matrices_from_fold_history(raw_data, live_status)

        _apply_lookback_memory(raw_data, live_status)

        cached_advice = raw_data.get("searchSpaceAdvice")
        if not _has_usable_search_space_advice(cached_advice):
            try:
                advisor = _get_search_space_advisor()
                dataset_fingerprint = None
                dataset_profile = None
                try:
                    dataset_fingerprint, dataset_profile = compute_dataset_profile_fingerprint()
                except Exception as e:
                    _log_event(
                        "warning",
                        f"Search space dataset profiling failed: {e}",
                        key_parameters={},
                        stop_reason="search_space_dataset_profile_failed",
                    )
                raw_data["searchSpaceAdvice"] = advisor.advise(
                    search_space=raw_data.get("searchSpace", {}),
                    trials_data=raw_data.get("trials", []),
                    importances=raw_data.get("importances", {}),
                    direction=raw_data.get("direction", "maximize"),
                    study_name=raw_data.get("studyName", ""),
                    dataset_fingerprint=dataset_fingerprint,
                    dataset_profile=dataset_profile,
                    objective_directions=(
                        raw_data.get("objectiveDirections")
                        if isinstance(raw_data.get("objectiveDirections"), list)
                        else None
                    ),
                    advisor_config=(
                        raw_data.get("searchSpaceCoverage")
                        if isinstance(raw_data.get("searchSpaceCoverage"), dict)
                        else None
                    ),
                )
            except Exception as e:
                _log_event(
                    "warning",
                    f"Search space advice recomputation failed: {e}",
                    key_parameters={},
                    stop_reason="search_space_advice_recompute_failed",
                )

        return raw_data

    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        if len(args) >= 2:
            status = args[1]
            if isinstance(status, str) and status.startswith("2"):
                return

        try:
            message = format % args
        except Exception:
            message = f"{format} {args}"

        _log_event(
            "debug",
            message,
            key_parameters={"path": self.path},
            stop_reason="http_log",
        )


def watchdog(parent_pid: int):
    """PID Watchdog: kills server if parent process terminates."""
    _log_event(
        "debug",
        f"Monitoring parent PID {parent_pid}",
        key_parameters={"parent_pid": parent_pid},
        stop_reason="watchdog_started",
    )

    build_pids: list[int] = []

    while not should_stop():
        try:
            os.kill(parent_pid, 0)
        except OSError:
            _log_event(
                "warning",
                f"Parent process {parent_pid} died. Initiating graceful shutdown",
                key_parameters={"parent_pid": parent_pid},
                stop_reason="parent_died",
            )

            for pid in build_pids:
                try:
                    os.kill(pid, 9)
                    _log_event(
                        "debug",
                        f"Killed build process {pid}",
                        key_parameters={"pid": pid},
                        stop_reason="build_process_killed",
                    )
                except OSError:
                    pass

            get_interrupt_manager().force_stop(reason="Parent process died")
            break

        time.sleep(2)


def run_server(port: int = 8766, parent_pid: int | None = None, bind: str = "0.0.0.0"):
    """Starts the dashboard server (programmatic entrypoint)."""
    global logger
    logger = _get_dashboard_logger()
    if parent_pid:
        t = threading.Thread(target=watchdog, args=(parent_pid,), daemon=True)
        t.start()

    _log_event(
        "info",
        f"Peak State Dashboard Server em http://{bind}:{port}",
        key_parameters={"bind": bind, "port": port},
        stop_reason="startup",
    )
    _log_event(
        "info",
        f"Servindo de: {DASHBOARD_DIR}",
        key_parameters={"path": str(DASHBOARD_DIR)},
        stop_reason="startup",
    )
    _log_event(
        "info",
        f"Arquivos estaticos: {STATIC_DIR}",
        key_parameters={"path": str(STATIC_DIR)},
        stop_reason="startup",
    )
    _log_event(
        "info",
        f"Pre-compilado: {DIST_DIR}",
        key_parameters={"path": str(DIST_DIR)},
        stop_reason="startup",
    )

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer.daemon_threads = True

    with socketserver.ThreadingTCPServer((bind, port), PeakStateDashboardHandler) as httpd:
        httpd.daemon_threads = True

        prev_sigterm = None
        prev_sigint = None

        if threading.current_thread() is threading.main_thread():

            def _handle_signal(signum: int, _frame):
                """Execute handle signal.



                Args:

                    signum: Input value used by this callable.

                    _frame: Input value used by this callable.

                """

                _log_event(
                    "warning",
                    f"Received signal {signum}; shutting down server",
                    key_parameters={"signal": signum},
                    stop_reason="signal_received",
                )
                get_interrupt_manager().force_stop(reason=f"signal_{signum}")
                try:
                    httpd.shutdown()
                except Exception:
                    pass

            prev_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
            prev_sigint = signal.signal(signal.SIGINT, _handle_signal)

        server_thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.5},
            name="hpo_dashboard_serve_forever",
            daemon=True,
        )
        server_thread.start()

        get_interrupt_manager().register_callback(
            lambda: httpd.shutdown(), label="hpo_dashboard_server_stop"
        )

        try:
            while server_thread.is_alive() and not should_stop():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                httpd.shutdown()
            except Exception:
                pass
            server_thread.join(timeout=2.0)

            if prev_sigterm is not None:
                signal.signal(signal.SIGTERM, prev_sigterm)
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)
            _log_event(
                "success",
                "Dashboard interrompido com sucesso.",
                key_parameters={},
                stop_reason="shutdown",
            )

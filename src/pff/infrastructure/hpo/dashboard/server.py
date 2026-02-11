"""Peak State HPO Dashboard Server.

Implements:
1. Strict Cache-Control:
    - index.html: no-cache, must-revalidate
    - /dist/*: public, max-age=31536000, immutable
2. Multi-path Data Loader: fallbacks between cache/ and outputs/
3. Lookback Logic: holds last valid validation metrics if current epoch is train-only.
4. PID Watchdog: self-terminates if parent dies.
"""

import argparse
import http.server
import json
import os
import re
import signal
import socketserver
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from collections.abc import Callable

from pff.shared.core.file_manager import FileManager
from pff.shared.ops.global_interrupt_manager import get_interrupt_manager, should_stop
from pff.shared.acceleration.concurrency import HardwareManager
from pff.shared.core.config import settings
from pff.shared.core.logging import LOG_DIR, create_isolated_logger, logger
from pff.infrastructure.hpo.config_loader import load_live_plot_settings

DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_DIR = DASHBOARD_DIR / "static"
DIST_DIR = DASHBOARD_DIR / "dist"
BASE_DIR = settings.ROOT_DIR
_DASHBOARD_LOGGER = None
DATA_CACHE_PATH: Path | None = None
_DATA_PATHS_CACHE: dict[str, Any] = {
    "paths": tuple(),
    "last_refresh": 0.0,
    "cache_root_mtime": None,
    "output_subdir": None,
    "data_cache_path": None,
}
_DATA_PATHS_CACHE_TTL_S = 1.0
_TELEMETRY_CACHE: dict[str, Any] = {"value": None, "last_refresh": 0.0}
_TELEMETRY_CACHE_TTL_S = 1.0
_HARDWARE_HISTORY: dict[str, Any] = {"items": [], "last_id": 0}


def _get_dashboard_logger():
    global _DASHBOARD_LOGGER
    if _DASHBOARD_LOGGER is None:
        _DASHBOARD_LOGGER = create_isolated_logger(
            "hpo_dashboard", log_dir=LOG_DIR / "dashboard"
        )
    return _DASHBOARD_LOGGER


def _resolve_dashboard_data_path(live_cfg: dict[str, Any] | None = None) -> Path:
    if DATA_CACHE_PATH is not None:
        return DATA_CACHE_PATH
    cfg = live_cfg or load_live_plot_settings()
    data_path = cfg.get("dashboard_data_path") if isinstance(cfg, dict) else None
    if data_path:
        resolved = Path(data_path)
        if not resolved.is_absolute():
            resolved = settings.ROOT_DIR / resolved
        return resolved
    return settings.CACHE_DIR / "hpo" / "dashboard_data.json"


def _reset_dashboard_paths_cache() -> None:
    _DATA_PATHS_CACHE["paths"] = tuple()
    _DATA_PATHS_CACHE["last_refresh"] = 0.0
    _DATA_PATHS_CACHE["cache_root_mtime"] = None
    _DATA_PATHS_CACHE["output_subdir"] = None
    _DATA_PATHS_CACHE["data_cache_path"] = None


def _reset_telemetry_cache() -> None:
    _TELEMETRY_CACHE["value"] = None
    _TELEMETRY_CACHE["last_refresh"] = 0.0


def _append_hardware_history(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
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
    now = time.time()

    cached_paths = _DATA_PATHS_CACHE["paths"]
    if (
        cached_paths
        and now - _DATA_PATHS_CACHE["last_refresh"] < _DATA_PATHS_CACHE_TTL_S
    ):
        return list(cached_paths)

    live_cfg = load_live_plot_settings()
    data_cache_path = _resolve_dashboard_data_path(live_cfg)
    output_subdir = live_cfg.get("output_subdir", "optimization/plots")
    live_plot_dir = settings.OUTPUTS_DIR / Path(output_subdir)
    cache_root = settings.CACHE_DIR / "hpo"
    cache_root_mtime = cache_root.stat().st_mtime if cache_root.exists() else None

    if (
        cached_paths
        and _DATA_PATHS_CACHE["output_subdir"] == output_subdir
        and _DATA_PATHS_CACHE["data_cache_path"] == data_cache_path
        and _DATA_PATHS_CACHE["cache_root_mtime"] == cache_root_mtime
    ):
        _DATA_PATHS_CACHE["last_refresh"] = now
        return list(cached_paths)

    candidates = [
        BASE_DIR / "outputs" / "dashboard_data.json",
        data_cache_path,
        BASE_DIR / ".cache" / "hpo" / "dashboard_data.json",
        DASHBOARD_DIR / "dashboard_data.json",
        live_plot_dir / "dashboard_data.json",
    ]

    if cache_root.exists():
        candidates.extend(list(cache_root.rglob("dashboard_data.json")))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    _DATA_PATHS_CACHE["paths"] = tuple(unique)
    _DATA_PATHS_CACHE["last_refresh"] = now
    _DATA_PATHS_CACHE["output_subdir"] = output_subdir
    _DATA_PATHS_CACHE["data_cache_path"] = data_cache_path
    _DATA_PATHS_CACHE["cache_root_mtime"] = cache_root_mtime
    return unique


def _log_event(
    level: str,
    message: str,
    *,
    key_parameters: dict[str, Any] | None = None,
    stop_reason: str = "none",
) -> None:
    bound = logger.bind(
        component="hpo_dashboard",
        key_parameters=key_parameters or {},
        stop_reason=stop_reason,
    )
    getattr(bound, level)(message)


def _get_cached_telemetry(hardware_manager: HardwareManager) -> dict[str, Any]:
    now = time.time()
    cached = _TELEMETRY_CACHE["value"]
    if (
        cached is not None
        and now - _TELEMETRY_CACHE["last_refresh"] < _TELEMETRY_CACHE_TTL_S
    ):
        return cast(dict[str, Any], cached)
    telemetry: dict[str, Any] = hardware_manager.get_telemetry()
    _TELEMETRY_CACHE["value"] = telemetry
    _TELEMETRY_CACHE["last_refresh"] = now
    return telemetry


def _read_tail_lines(
    path: Path, *, max_bytes: int = 65536, max_lines: int = 200
) -> list[str]:
    if max_bytes <= 0 or max_lines <= 0:
        return []
    raw = FileManager.read_tail_bytes(path, max_bytes=max_bytes)
    if not raw:
        return []
    text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines:
        return []
    return lines[-max_lines:]


# Patterns to suppress from the dashboard log viewer (known, non-actionable).
_LOG_SUPPRESSION_PATTERNS: list[str] = [
    "numpy_compat_shim",
    "NumPy compatibility shim",
]

_LOGURU_PIPE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s*\|\s*"
    r"(?P<level>\w+)\s*\|\s*"
    r"(?P<module>[^\|]+?)\s*\|"
    r".*?\|\s*"
    r"(?:component_name=\S+\s+)?(?:stop_reason=\S+\s+)?(?:key_parameters=\{[^}]*\}\s+)?"
    r"message='(?P<message>[^']+)'"
)


def _parse_log_line(raw: str) -> dict[str, str] | None:
    """Parse a Loguru readable-format line into structured fields.

    Returns None when the line should be suppressed or is unparseable.
    """
    stripped = raw.strip()
    if not stripped:
        return None

    # Suppress known noise patterns
    for pattern in _LOG_SUPPRESSION_PATTERNS:
        if pattern in stripped:
            return None

    # Try structured Loguru format first
    m = _LOGURU_PIPE_RE.match(stripped)
    if m:
        return {
            "timestamp": m.group("ts"),
            "level": m.group("level").strip().upper(),
            "module": m.group("module").strip().rsplit(".", 1)[-1],
            "message": m.group("message").strip(),
        }

    # Fallback: try simpler pipe-delimited extraction
    parts = stripped.split("|")
    if len(parts) >= 3:
        ts = parts[0].strip()
        level = parts[1].strip().upper()
        # Extract message from the structured portion after pipes
        rest = "|".join(parts[3:])  # Skip module (parts[2])
        # Try to find message='...' in rest
        msg_match = re.search(r"message='([^']+)'", rest)
        if msg_match:
            msg = msg_match.group(1).strip()
        else:
            # Lines without message='...' (e.g. interrupt manager):
            # TS | LEVEL | MODULE | task=X | ... | stop=X | ACTUAL_MSG | params={}
            # Strip metadata segments and params, keep human-readable portion
            segments = [s.strip() for s in rest.split("|")]
            # Remove task=, trace=, span=, stop=, params= segments
            human_parts = [
                s
                for s in segments
                if s
                and not re.match(
                    r"^(task=|trace=|span=|stop=|params=|component_name=|"
                    r"stop_reason=|key_parameters=)",
                    s,
                )
            ]
            msg = " ".join(human_parts).strip() if human_parts else rest.strip()
        module = parts[2].strip().rsplit(".", 1)[-1] if len(parts) > 2 else ""
        if ts and level in ("ERROR", "WARNING", "CRITICAL"):
            return {
                "timestamp": ts,
                "level": level,
                "module": module,
                "message": msg,
            }

    # JSON payload (legacy format)
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                text = payload.get("text")
                if isinstance(text, str) and text.strip():
                    return {
                        "timestamp": "",
                        "level": "WARNING",
                        "module": "",
                        "message": text.strip(),
                    }
        except Exception:
            pass

    # Last resort: raw string as warning
    return {
        "timestamp": "",
        "level": "WARNING",
        "module": "",
        "message": stripped,
    }


def _normalize_log_entries(lines: list[str]) -> list[dict[str, str]]:
    """Parse raw log lines into structured entries for the dashboard."""
    entries: list[dict[str, str]] = []
    for line in lines:
        entry = _parse_log_line(line)
        if entry is not None:
            entries.append(entry)
    return entries


_MAX_LOG_ENTRIES = 200
_MAX_TAIL_BYTES = 65536
_MAX_TAIL_LINES = 150

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


def _epoch_score(item: dict[str, Any]) -> float:
    """Extracts the best available score from an epoch metrics dict."""
    raw = (
        item.get("score")
        or item.get("mrr")
        or item.get("mcc")
        or item.get("accuracy")
        or 0.0
    )
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


def _load_raw_dashboard_data() -> dict[str, Any]:
    """Reads the most recent dashboard JSON from multi-path candidates."""
    paths = _collect_dashboard_data_paths()
    valid_files = [p for p in paths if FileManager.exists(p)]
    if not valid_files:
        return {}
    newest = max(valid_files, key=lambda p: p.stat().st_mtime)
    try:
        return cast(dict[str, Any], FileManager.read(newest, return_native=True))
    except Exception as e:
        _log_event(
            "warning",
            f"Failed to load dashboard data: {e}",
            key_parameters={"path": str(newest)},
            stop_reason="dashboard_data_read_failed",
        )
        return {}


def _load_live_status() -> dict[str, Any] | None:
    """Reads the live_status.json produced by the training loop."""
    path = BASE_DIR / "outputs" / "optimization" / "plots" / "live_status.json"
    if not FileManager.exists(path):
        return None
    try:
        data: dict[str, Any] = cast(
            dict[str, Any], FileManager.read(path, return_native=True)
        )
        return data
    except Exception:
        return None


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
            chunk = _read_tail_lines(
                path, max_bytes=_MAX_TAIL_BYTES, max_lines=_MAX_TAIL_LINES
            )
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
        live_status.setdefault("gpu_utilization", telemetry["gpus"][0]["utilization"])
        live_status.setdefault(
            "vram_utilization", telemetry["gpus"][0]["vram_usage_pct"]
        )
    live_status.setdefault("ram_utilization", telemetry["ram_usage_pct"])
    return live_status


def _apply_study_defaults(raw_data: dict[str, Any]) -> None:
    """Sets study-level defaults on raw_data from HPO configuration."""
    study_name = raw_data.get("studyName")
    if not study_name or study_name == "Initializing...":
        study_name = settings.HPO_CONFIG.get("study_name", "PFF HPO Study")
    hpo_defaults = settings.HPO_CONFIG.get("defaults", {})
    raw_data.setdefault("studyName", study_name)
    raw_data.setdefault("direction", hpo_defaults.get("direction", "maximize"))
    raw_data.setdefault("totalTrials", hpo_defaults.get("n_trials", 50))
    raw_data.setdefault("charts", {})
    raw_data.setdefault("updatedAt", datetime.now(timezone.utc).isoformat())
    raw_data.setdefault("_synthetic_trials", False)


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
            debug_status["trial_number"] = (
                max(v for v in valid_ids if v is not None) - 1
            )
    raw_data["updatedAt"] = debug_status["updated_at"]
    return debug_status


def _consolidate_live_trial(
    raw_data: dict[str, Any],
    live_status: dict[str, Any] | None,
) -> None:
    """Merges current running trial data from live_status into raw_data trials."""
    if not isinstance(live_status, dict) or "trial_number" not in live_status:
        return

    debug_mode = raw_data.get("dashboardDebugMode", False)
    valid_files_exist = any(p.exists() for p in _collect_dashboard_data_paths())

    try:
        trial_val = live_status.get("trial_number")
        if trial_val is None or not isinstance(trial_val, (int, float)):
            return

        live_id = trial_val
        synthetic_id = -abs(int(live_id)) if int(live_id) != 0 else -1
        trials_list = raw_data.get("trials", [])
        if not isinstance(trials_list, list):
            trials_list = []

        trials_map = {}
        for t in trials_list:
            if not isinstance(t, dict):
                continue
            tid = t.get("id")
            if tid is None:
                continue
            if isinstance(tid, (int, float)):
                tid = abs(int(tid))
                t["id"] = tid
            if tid == synthetic_id:
                continue
            trials_map[tid] = t

        epoch_history = live_status.get("epoch_history", [])
        best_epoch_metrics, best_epoch_score = _extract_best_epoch(epoch_history)

        if isinstance(raw_data.get("liveStatus"), dict):
            raw_data["liveStatus"]["trial_number"] = live_id

        live_row = _find_live_row(trials_map, live_id)

        if live_row:
            best_epoch_metrics = _update_lookback_best(
                live_id, live_row, best_epoch_metrics
            )

        if live_row and best_epoch_metrics and live_row.get("state") != "COMPLETE":
            _merge_epoch_into_trial(
                live_row, best_epoch_metrics, best_epoch_score, live_status
            )

        if not valid_files_exist and live_id not in trials_map and not debug_mode:
            _create_synthetic_trial(trials_map, synthetic_id, live_status, raw_data)

        raw_data["trials"] = sorted(
            trials_map.values(), key=lambda x: int(x.get("id", 0))
        )

    except (ValueError, TypeError) as e:
        _log_event(
            "warning",
            f"Failed to consolidate live data: {e}",
            key_parameters={"study": raw_data.get("studyName")},
            stop_reason="live_data_consolidation_failed",
        )


def _extract_best_epoch(
    epoch_history: list,
) -> tuple[dict[str, Any] | None, float]:
    """Finds the best epoch metrics from the epoch history."""
    if not isinstance(epoch_history, list) or not epoch_history:
        return None, 0.0
    metrics_list = [e for e in epoch_history if isinstance(e, dict)]
    if not metrics_list:
        return None, 0.0
    scored_epochs = [e for e in metrics_list if _has_metrics(e)]
    best = max(scored_epochs, key=_epoch_score) if scored_epochs else metrics_list[-1]
    return best, _epoch_score(best)


def _find_live_row(trials_map: dict, live_id: float) -> dict[str, Any] | None:
    """Finds the trial row matching the current live trial."""
    try:
        base_id = int(live_id)
        candidate_ids = [base_id, base_id + 1]
    except (TypeError, ValueError):
        return None

    live_row = None
    for cid in candidate_ids:
        candidate = trials_map.get(cid)
        if not candidate:
            continue
        if candidate.get("state") != "COMPLETE":
            live_row = candidate
            break
        if live_row is None:
            live_row = candidate
    return live_row


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

    with LOOKBACK_LOCK:
        live_best = LOOKBACK_MEMORY.get("live_best_metrics")
        if not isinstance(live_best, dict):
            live_best = {}
            LOOKBACK_MEMORY["live_best_metrics"] = live_best

        previous_best = live_best.get(live_key, {})
        if best_epoch_metrics:
            cleaned = {k: v for k, v in best_epoch_metrics.items() if v is not None}
            merged = {**previous_best, **cleaned}
            live_best[live_key] = merged
            best_epoch_metrics = merged
        elif previous_best:
            best_epoch_metrics = previous_best

    if live_row.get("state") == "COMPLETE":
        with LOOKBACK_LOCK:
            live_best = LOOKBACK_MEMORY.get("live_best_metrics")
            if isinstance(live_best, dict):
                live_best.pop(live_key, None)

    return best_epoch_metrics


def _merge_epoch_into_trial(
    live_row: dict[str, Any],
    best_epoch_metrics: dict[str, Any],
    best_epoch_score: float,
    live_status: dict[str, Any],
) -> None:
    """Merges best epoch metrics into the live trial row."""
    loss_value = (
        best_epoch_metrics.get("loss")
        or best_epoch_metrics.get("val_loss")
        or best_epoch_metrics.get("train_loss")
        or best_epoch_metrics.get("binary_loss")
    )
    live_status_payload = live_status if isinstance(live_status, dict) else {}
    duration = float(live_status_payload.get("elapsed_seconds", 0.0) or 0.0)
    efficiency = None
    if duration:
        try:
            efficiency = float(best_epoch_score) / duration
        except (TypeError, ValueError):
            efficiency = None

    _raw_metrics = live_row.get("metrics")
    metrics_payload: dict[str, Any] = _raw_metrics if isinstance(_raw_metrics, dict) else {}  # type: ignore
    metrics_payload = {**metrics_payload, **_clean_metrics(best_epoch_metrics)}
    if duration:
        metrics_payload.setdefault("duration", duration)
    if efficiency is not None:
        metrics_payload.setdefault("efficiency", efficiency)
    if loss_value is not None:
        metrics_payload.setdefault("loss", loss_value)

    update_payload: dict[str, Any] = {"metrics": metrics_payload}
    if loss_value is not None:
        update_payload["loss"] = loss_value
    if efficiency is not None:
        update_payload["efficiency"] = efficiency
    if duration:
        update_payload["duration"] = duration

    for key in ("precision", "recall", "mrr", "mcc", "accuracy", "f1", "auc"):
        val = best_epoch_metrics.get(key)
        if val is not None:
            update_payload[key] = val

    for orig, norm in (("hits@1", "hits1"), ("hits@3", "hits3"), ("hits@10", "hits10")):
        hit_val = best_epoch_metrics.get(orig)
        if hit_val is None:
            hit_val = best_epoch_metrics.get(norm)
        if hit_val is not None:
            update_payload[norm] = hit_val

    live_row.update(update_payload)


def _create_synthetic_trial(
    trials_map: dict,
    synthetic_id: int,
    live_status: dict[str, Any],
    raw_data: dict[str, Any],
) -> None:
    """Creates a synthetic RUNNING trial entry when no dashboard files exist."""
    epoch_history = live_status.get("epoch_history", [])
    live_score = 0.0
    last_metrics: dict[str, Any] = {}
    best_metrics = {"mrr": 0.0, "mcc": 0.0}

    if isinstance(epoch_history, list) and epoch_history:
        metrics_list = [e for e in epoch_history if isinstance(e, dict)]
        if metrics_list:
            live_score = max(
                float(e.get("mrr") or e.get("score") or 0.0) for e in metrics_list
            )
            last_metrics = metrics_list[-1]
            best_metrics["mrr"] = max(float(e.get("mrr", 0.0)) for e in metrics_list)
            best_metrics["mcc"] = max(float(e.get("mcc", 0.0)) for e in metrics_list)

    loss_value = (
        last_metrics.get("loss")
        or last_metrics.get("val_loss")
        or last_metrics.get("train_loss")
        or last_metrics.get("binary_loss")
    )
    duration = float(live_status.get("elapsed_seconds", 0.0) or 0.0)
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
        live_status.get("warmstart") or live_status.get("warmstart_seed")
    )

    trials_map[synthetic_id] = {
        "id": synthetic_id,
        "value": live_score,
        "state": "RUNNING",
        "params": live_status.get("params", {}),
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
    existing = LOOKBACK_MEMORY.get("confusion_matrices")
    history = existing if isinstance(existing, list) else []
    if (
        history
        and history[-1].get("cv_fold_id") == cv_fold_id
        and history[-1].get("trial_number") == trial_number
    ):
        history[-1] = entry
    else:
        history = history + [entry]
    LOOKBACK_MEMORY["confusion_matrices"] = history[-3:]


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
        charts.get("gen_gap")
        or charts.get("confusion_matrix")
        or charts.get("confusion_matrices")
    )

    with LOOKBACK_LOCK:
        if has_fresh_validation:
            LOOKBACK_MEMORY["gen_gap"] = charts.get("gen_gap")
            LOOKBACK_MEMORY["confusion_matrix"] = charts.get("confusion_matrix")
            _update_fold_memory(charts.get("confusion_matrix"), current)
            if charts.get("confusion_matrices"):
                LOOKBACK_MEMORY["confusion_matrices"] = charts.get("confusion_matrices")
            LOOKBACK_MEMORY["last_valid_epoch"] = current.get("current_epoch", -1)
            LOOKBACK_MEMORY["source_trial"] = current.get("trial_number", -1)
            raw_data["stale_validation"] = False
        else:
            raw_data["stale_validation"] = True
            if LOOKBACK_MEMORY["gen_gap"]:
                charts["gen_gap"] = LOOKBACK_MEMORY["gen_gap"]
            if LOOKBACK_MEMORY["confusion_matrix"]:
                charts["confusion_matrix"] = LOOKBACK_MEMORY["confusion_matrix"]
            if LOOKBACK_MEMORY["confusion_matrices"]:
                charts["confusion_matrices"] = LOOKBACK_MEMORY["confusion_matrices"]
            charts["lookback_epoch"] = LOOKBACK_MEMORY["last_valid_epoch"]
            charts["lookback_trial"] = LOOKBACK_MEMORY["source_trial"]


LOOKBACK_MEMORY: dict[str, Any] = {
    "gen_gap": None,
    "confusion_matrix": None,
    "confusion_matrices": None,
    "last_valid_epoch": -1,
    "source_trial": -1,
    "live_best_metrics": {},
}

LOOKBACK_LOCK = threading.Lock()


# ── Export handlers (Strategy pattern: Replace Conditional with Polymorphism) ──


def _export_csv(data: dict[str, Any], **_kw: Any) -> tuple[bytes, str]:
    """Exports trial data as CSV."""
    import csv
    import io

    output = io.StringIO()
    trials = data.get("trials", [])
    if trials:
        keys: set[str] = set()
        for t in trials:
            keys.update(t.keys())
            if "params" in t:
                keys.update(f"param_{k}" for k in t["params"].keys())
            if "metrics" in t:
                keys.update(f"metric_{k}" for k in t["metrics"].keys())

        writer = csv.writer(output)
        header = sorted(keys)
        writer.writerow(header)
        for t in trials:
            row = []
            for k in header:
                if k.startswith("param_"):
                    val = t.get("params", {}).get(k[6:])
                elif k.startswith("metric_"):
                    val = t.get("metrics", {}).get(k[7:])
                else:
                    val = t.get(k)
                row.append(val)
            writer.writerow(row)

    return output.getvalue().encode("utf-8"), "text/csv"


def _export_parquet(data: dict[str, Any], **_kw: Any) -> tuple[bytes, str]:
    """Exports trial data as Parquet."""
    import io

    import polars as pl

    trials = data.get("trials", [])
    flattened = []
    for t in trials:
        row = {k: v for k, v in t.items() if k not in ("params", "metrics")}
        if "params" in t:
            row.update({f"param_{k}": v for k, v in t["params"].items()})
        if "metrics" in t:
            row.update({f"metric_{k}": v for k, v in t["metrics"].items()})
        flattened.append(row)

    df = pl.DataFrame(flattened)
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    return buffer.getvalue(), "application/octet-stream"


def _export_toon(data: dict[str, Any], **_kw: Any) -> tuple[bytes, str]:
    """Exports trial summary as PFF-branded ASCII art report."""
    trials = data.get("trials", [])
    study_name = data.get("studyName", "Unknown Study")
    direction = data.get("direction", "maximize")

    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        f"║ PFF :: HPO EXPORT :: {study_name.upper():<31} ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Direction: {direction}",
        f"  Total Trials: {len(trials)}",
        "",
        "  [ TOP TRIALS ]",
    ]

    valid_trials = [t for t in trials if t.get("value") is not None]
    top_trials = sorted(
        valid_trials,
        key=lambda x: x["value"],
        reverse=(direction == "maximize"),
    )[:5]

    for i, t in enumerate(top_trials):
        lines.append(f"  {i + 1}. Trial #{t['id']}: {t['value']:.6f} ({t['state']})")
        if t.get("params"):
            params_str = ", ".join(f"{k}={v}" for k, v in t["params"].items())
            lines.append(f"     Params: {params_str[:80]}...")

    lines.append("")
    lines.append("  [ RAW DATA ]")
    lines.append(
        "  " + "ID".ljust(6) + "VALUE".ljust(12) + "STATE".ljust(12) + "DURATION"
    )
    lines.append("  " + "-" * 40)

    for t in trials[-20:]:
        tid = str(t["id"]).ljust(6)
        val = str(round(t.get("value", 0), 4)).ljust(12)
        state = t["state"].ljust(12)
        dur = f"{t.get('duration', 0):.1f}s"
        lines.append(f"  {tid}{val}{state}{dur}")

    lines.append("╚══════════════════════════════════════════════════════════════╝")
    return "\n".join(lines).encode("utf-8"), "text/plain"


def _export_json(data: dict[str, Any], **_kw: Any) -> tuple[bytes, str]:
    """Exports raw data as JSON (default fallback)."""
    content = FileManager.json_dumps(data, sort_keys=True).encode("utf-8")
    return content, "application/json"


_EXPORT_HANDLERS: dict[str, Callable[..., tuple[bytes, str]]] = {
    "csv": _export_csv,
    "parquet": _export_parquet,
    "toon": _export_toon,
    "json": _export_json,
}


class PeakStateDashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for Peak State performance and robustness."""

    protocol_version = "HTTP/1.1"
    hardware_manager: HardwareManager | None = None

    def __init__(self, *args, **kwargs):
        if self.__class__.hardware_manager is None:
            self.__class__.hardware_manager = HardwareManager()
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

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
                    FileManager.read_bytes(build_id_path)
                    .decode("utf-8", errors="ignore")
                    .strip()
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
                payload = json.loads(body)
                fmt = payload.get("format", "json")
                filename = payload.get("filename", "export")
                data = payload.get("data", {})

                handler = _EXPORT_HANDLERS.get(fmt, _export_json)
                content, content_type = handler(data)

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{filename}.{fmt}"'
                )
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
            file_path.endswith(".js") or file_path.endswith(".css")
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

        self.wfile.write(b": ping\n\n")
        self.wfile.flush()

        try:
            data = self._load_consolidated_data()
            payload = FileManager.json_dumps(data)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
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
                    max([p.stat().st_mtime for p in valid_files])
                    if valid_files
                    else 0.0
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
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
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
        if self.path.startswith("/api/status"):
            data = {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
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

    def _load_consolidated_data(self) -> dict[str, Any]:
        """Loads data from multi-path and applies lookback memory + live consolidation."""
        raw_data = _load_raw_dashboard_data()
        live_status = _load_live_status()

        raw_data.setdefault("trials", [])

        live_status = _collect_terminal_logs(live_status, raw_data)

        live_status = _inject_telemetry(self, live_status)

        raw_data["liveStatus"] = live_status

        _apply_study_defaults(raw_data)

        live_status = _apply_debug_mode(raw_data, live_status)

        _consolidate_live_trial(raw_data, live_status)

        _compute_best_value(raw_data)

        _apply_lookback_memory(raw_data, live_status)

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

    with socketserver.ThreadingTCPServer(
        (bind, port), PeakStateDashboardHandler
    ) as httpd:
        httpd.daemon_threads = True

        prev_sigterm = None
        prev_sigint = None

        if threading.current_thread() is threading.main_thread():

            def _handle_signal(signum: int, _frame):
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


def main():
    parser = argparse.ArgumentParser(description="Peak State HPO Dashboard Server")
    parser.add_argument("--port", type=int, default=8766, help="Server port")
    parser.add_argument("--bind", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument(
        "--parent-pid", type=int, default=None, help="Parent PID for watchdog"
    )
    args = parser.parse_args()

    run_server(port=args.port, parent_pid=args.parent_pid, bind=args.bind)


if __name__ == "__main__":
    main()

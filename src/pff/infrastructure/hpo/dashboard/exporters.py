"""Dashboard export handlers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from pff.shared.core.file_manager import FileManager


def normalize_direction_label(direction: Any) -> str:
    """Normalize objective direction to maximize/minimize."""
    raw = str(direction or "maximize").strip().lower()
    if "." in raw:
        raw = raw.split(".")[-1]
    if raw in {"maximize", "max"}:
        return "maximize"
    if raw in {"minimize", "min"}:
        return "minimize"
    return "maximize"


def export_csv(data: dict[str, Any], **_kw: Any) -> tuple[bytes, str]:
    """Export trial data as CSV."""
    import csv
    import io

    output = io.StringIO()
    trials = data.get("trials", [])
    if trials:
        keys: set[str] = set()
        for trial in trials:
            keys.update(trial.keys())
            if "params" in trial:
                keys.update(f"param_{k}" for k in trial["params"].keys())
            if "metrics" in trial:
                keys.update(f"metric_{k}" for k in trial["metrics"].keys())

        writer = csv.writer(output)
        header = sorted(keys)
        writer.writerow(header)
        for trial in trials:
            row = []
            for key in header:
                if key.startswith("param_"):
                    value = trial.get("params", {}).get(key[6:])
                elif key.startswith("metric_"):
                    value = trial.get("metrics", {}).get(key[7:])
                else:
                    value = trial.get(key)
                row.append(value)
            writer.writerow(row)

    return output.getvalue().encode("utf-8"), "text/csv"


def export_parquet(data: dict[str, Any], **_kw: Any) -> tuple[bytes, str]:
    """Export trial data as Parquet."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    trials = data.get("trials", [])
    flattened = []
    for trial in trials:
        row = {k: v for k, v in trial.items() if k not in ("params", "metrics")}
        if "params" in trial:
            row.update({f"param_{k}": v for k, v in trial["params"].items()})
        if "metrics" in trial:
            row.update({f"metric_{k}": v for k, v in trial["metrics"].items()})
        flattened.append(row)

    table = pa.Table.from_pylist(flattened) if flattened else pa.table({})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue(), "application/octet-stream"


def export_toon(data: dict[str, Any], **_kw: Any) -> tuple[bytes, str]:
    """Export trial summary as ASCII report."""
    trials = data.get("trials", [])
    study_name = data.get("studyName", "Unknown Study")
    direction = normalize_direction_label(data.get("direction", "maximize"))

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

    valid_trials = [trial for trial in trials if trial.get("value") is not None]
    top_trials = sorted(
        valid_trials,
        key=lambda item: item["value"],
        reverse=(direction == "maximize"),
    )[:5]

    for idx, trial in enumerate(top_trials):
        lines.append(
            f"  {idx + 1}. Trial #{trial['id']}: {trial['value']:.6f} ({trial['state']})"
        )
        if trial.get("params"):
            params_str = ", ".join(f"{k}={v}" for k, v in trial["params"].items())
            lines.append(f"     Params: {params_str[:80]}...")

    lines.append("")
    lines.append("  [ RAW DATA ]")
    lines.append(
        "  " + "ID".ljust(6) + "VALUE".ljust(12) + "STATE".ljust(12) + "DURATION"
    )
    lines.append("  " + "-" * 40)

    for trial in trials[-20:]:
        trial_id = str(trial["id"]).ljust(6)
        value = str(round(trial.get("value", 0), 4)).ljust(12)
        state = trial["state"].ljust(12)
        duration = f"{trial.get('duration', 0):.1f}s"
        lines.append(f"  {trial_id}{value}{state}{duration}")

    lines.append("╚══════════════════════════════════════════════════════════════╝")
    return "\n".join(lines).encode("utf-8"), "text/plain"


def export_json(data: dict[str, Any], **_kw: Any) -> tuple[bytes, str]:
    """Export raw payload as JSON."""
    content = FileManager.json_dumps(data, sort_keys=True).encode("utf-8")
    return content, "application/json"


EXPORT_HANDLERS: dict[str, Callable[..., tuple[bytes, str]]] = {
    "csv": export_csv,
    "parquet": export_parquet,
    "toon": export_toon,
    "json": export_json,
}


__all__ = [
    "EXPORT_HANDLERS",
    "export_csv",
    "export_json",
    "export_parquet",
    "export_toon",
    "normalize_direction_label",
]

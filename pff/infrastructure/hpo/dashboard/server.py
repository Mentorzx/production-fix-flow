"""Dashboard server for HPO live monitoring.

This module provides a simple HTTP server that serves the static dashboard
and provides a REST endpoint to fetch the latest HPO data.

Usage:
    python -m pff.infrastructure.hpo.dashboard.server [--port 8766]
"""

from __future__ import annotations

import json
import http.server
import socketserver
import argparse
import signal
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from pff import settings
from pff.shared import logger
from pff.shared.system.resource_manager import get_resource_manager

DASHBOARD_DIR = Path(__file__).parent
STATIC_DIR = DASHBOARD_DIR / "static"
DATA_CACHE_PATH = settings.CACHE_DIR / "hpo" / "dashboard_data.json"


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler that serves dashboard files and data API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        from urllib.parse import urlparse

        clean_path = urlparse(self.path).path
        if clean_path == "/api/data" or clean_path == "/api/data.json":
            self._serve_data_api()
        elif clean_path == "/api/status":
            self._serve_status_api()
        else:
            super().do_GET()

    def do_POST(self):
        """Handle POST requests (for data export)."""
        from urllib.parse import urlparse

        clean_path = urlparse(self.path).path
        if clean_path == "/api/export":
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len > 0:
                body = self.rfile.read(content_len)
                try:
                    payload = json.loads(body)
                    self._serve_export_api(payload)
                except Exception as e:
                    self._send_json_response({"error": str(e)}, status=400)
            else:
                self._send_json_response({"error": "Empty body"}, status=400)
        else:
            self.send_error(404, "Not Found")

    def _serve_data_api(self):
        """Serve the HPO data as JSON."""
        try:
            data = {}
            # 1. Try to load full study data
            if DATA_CACHE_PATH.exists():
                data = json.loads(DATA_CACHE_PATH.read_text())
            else:
                # Basic template if no study data yet
                data = {
                    "studyName": "Initializing...",
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "bestValue": 0,
                    "trials": [],
                }

            # 2. Try to load live status (intra-trial progress)
            status_path = _find_live_status_path()
            if status_path is not None and status_path.exists():
                live_status = json.loads(status_path.read_text())

                # Add hardware data from resource_manager
                try:
                    resource_manager = get_resource_manager()
                    current_resources = resource_manager.get_current_resources()

                    # Base hardware metrics
                    hardware = {
                        "memory_total_gb": round(current_resources["memory_total_gb"], 2),
                        "memory_available_gb": round(current_resources["memory_available_gb"], 2),
                        "memory_used_gb": round(current_resources["memory_used_gb"], 2),
                        "memory_percent": round(current_resources["memory_percent"], 2),
                        "cpu_count": current_resources["cpu_count"],
                        "cpu_percent": round(current_resources["cpu_percent"], 2),
                        "gpu_utilization": None,
                        "vram_utilization": None,
                        "ram_utilization": round(current_resources["memory_percent"], 2),
                    }

                    # Try to get CUDA memory and utilization
                    try:
                        from pff.shared.system.resource_manager import (
                            get_cuda_memory_info,
                        )
                        import pynvml

                        cuda_info = get_cuda_memory_info()
                        if cuda_info is not None:
                            hardware["vram_utilization"] = round(
                                (1 - cuda_info["free_ratio"]) * 100, 2
                            )
                            hardware["vram_total_gb"] = round(
                                cuda_info["total_bytes"] / (1024**3), 2
                            )
                            hardware["vram_used_gb"] = round(cuda_info["used_bytes"] / (1024**3), 2)

                        try:
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            hardware["gpu_utilization"] = float(util.gpu)
                        except Exception:
                            pass
                    except Exception:
                        pass

                    live_status["hardware"] = hardware
                except Exception as e:
                    logger.debug(f"Erro ao coletar dados de hardware: {e}")

                data["liveStatus"] = live_status
                # If studyName is generic, try to be more specific
                if data["studyName"] == "Initializing..." and "trial_number" in live_status:
                    data["studyName"] = f"Running Trial #{live_status['trial_number']}"
                data["updatedAt"] = _max_updated_at(
                    data.get("updatedAt"), live_status.get("updated_at")
                )

            self._send_json_response(data)
        except Exception as e:
            logger.debug(f"Error serving data API: {e}")
            self._send_json_response({"error": str(e)}, status=500)

    def _serve_export_api(self, payload: dict[str, Any]):
        """Convert payload to requested format and return file."""
        import io

        fmt = payload.get("format", "json").lower()
        filename = payload.get("filename", "hpo_export")
        raw_data = payload.get("data", {})

        # Normalize data structure to ensure it works for all formats
        if isinstance(raw_data, list):
            data = {
                "trials": raw_data,
                "studyName": "Exported Trials List",
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                "bestValue": 0,
            }
        else:
            data = raw_data

        try:
            if fmt == "json":
                content = json.dumps(data, indent=2).encode("utf-8")
                content_type = "application/json"
                ext = "json"

            elif fmt == "csv":
                import csv

                # Extract trials list safely
                trials = data.get("trials", [])
                if not isinstance(trials, list):
                    trials = [data] if data else []

                if not trials:
                    content = b""
                else:
                    # Collect all keys
                    all_keys = set()
                    flat_rows = []
                    for t in trials:
                        if not isinstance(t, dict):
                            continue

                        row = {}
                        params = t.get("params", {})
                        metrics = t.get("metrics", {})

                        # Copy base keys (id, state, value, duration, etc.)
                        for k, v in t.items():
                            if k not in ("params", "metrics") and not isinstance(v, (dict, list)):
                                row[k] = v
                                all_keys.add(k)

                        if isinstance(params, dict):
                            for k, v in params.items():
                                key = f"param_{k}"
                                row[key] = v
                                all_keys.add(key)

                        if isinstance(metrics, dict):
                            for k, v in metrics.items():
                                row[k] = v
                                all_keys.add(k)

                        flat_rows.append(row)

                    # Sort keys: id first, then value, then others
                    sorted_keys = sorted(list(all_keys))
                    if "id" in all_keys:
                        sorted_keys.remove("id")
                        sorted_keys.insert(0, "id")

                    buffer = io.StringIO()
                    writer = csv.DictWriter(buffer, fieldnames=sorted_keys)
                    writer.writeheader()
                    writer.writerows(flat_rows)

                    content = buffer.getvalue().encode("utf-8")

                content_type = "text/csv"
                ext = "csv"

            elif fmt == "parquet":
                try:
                    import polars as pl
                except ImportError:
                    raise ImportError(
                        "Polars is required for Parquet export. Please install 'polars'."
                    )

                trials = data.get("trials", [])
                if not trials or not isinstance(trials, list):
                    raise ValueError("No trials data to export to Parquet")

                flat_rows = []
                for t in trials:
                    row = t.copy()
                    params = row.pop("params", {})
                    metrics = row.pop("metrics", {})
                    for k, v in params.items():
                        row[f"param_{k}"] = v
                    for k, v in metrics.items():
                        if k not in row:
                            row[k] = v
                    flat_rows.append(row)

                df = pl.DataFrame(flat_rows)
                buffer = io.BytesIO()
                df.write_parquet(buffer)
                content = buffer.getvalue()
                content_type = "application/octet-stream"
                ext = "parquet"

            elif fmt == "toon":
                # Token-Oriented Object Notation (Custom Readable Text)
                lines = []
                lines.append(f"STUDY: {data.get('studyName', 'Unknown')}")
                lines.append(f"UPDATED: {data.get('updatedAt', 'N/A')}")
                lines.append(f"BEST_VALUE: {data.get('bestValue', 0)}")

                if "sampler" in data:
                    lines.append(f"SAMPLER: {data.get('sampler')}")
                if "objectiveName" in data:
                    lines.append(
                        f"OBJECTIVE: {data.get('objectiveName')} ({data.get('direction', 'maximize')})"
                    )

                # Search Space
                ss = data.get("searchSpace", {})
                if ss:
                    lines.append("-" * 40)
                    lines.append("SEARCH_SPACE:")
                    for name, dist in ss.items():
                        lines.append(f"  {name}: {dist}")

                # Param Importances
                imp = data.get("importances", {})
                if imp:
                    lines.append("-" * 40)
                    lines.append("PARAMETER_IMPORTANCES (fANOVA):")
                    # Sort by importance descending
                    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
                    for name, val in sorted_imp:
                        lines.append(f"  {name}: {val:.4f} ({val * 100:.1f}%)")

                # Expanded Projections & Analysis
                proj = data.get("projections", {})
                if proj:
                    lines.append("-" * 40)
                    lines.append("ANALYSIS & PROJECTIONS:")
                    lines.append("  LINEAR_REGRESSION:")
                    lines.append(f"    SLOPE: {proj.get('slope', 0):.6f} / trial")
                    lines.append(f"    INTERCEPT: {proj.get('intercept', 0):.6f}")
                    lines.append(f"    VARIANCE: {proj.get('variance', 0):.6f}")
                    lines.append(f"    PREDICTED_VALUE_AT_END: {proj.get('predictedValue', 0):.4f}")

                # Hardware Info
                live = data.get("liveStatus", {})
                hw = live.get("hardware", {})
                if not hw:
                    hw = data.get("hardware", {})  # Fallback if direct in data

                if hw:
                    lines.append("-" * 40)
                    lines.append("HARDWARE_SNAPSHOT:")
                    for k, v in hw.items():
                        if v is not None:
                            lines.append(f"  {k.upper()}: {v}")

                # Live Trial Progress (The "In-Flight" data)
                if live and live.get("epoch_history"):
                    lines.append("-" * 40)
                    lines.append(
                        f"LIVE_TRIAL_PROGRESS (Trial #{live.get('trial_number', '?') + 1}):"
                    )
                    lines.append(
                        f"  CURRENT_EPOCH: {live.get('current_epoch')} / {live.get('total_epochs')}"
                    )
                    lines.append(f"  ELAPSED: {live.get('elapsed_seconds', 0):.1f}s")

                    history = live.get("epoch_history", [])
                    if history:
                        lines.append("  RECENT_EPOCHS:")
                        # Show last 5 epochs
                        for e in history[-5:]:
                            metrics_str = ", ".join(
                                [
                                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in e.items()
                                    if k != "epoch"
                                ]
                            )
                            lines.append(f"    Epoch {e.get('epoch')}: {metrics_str}")

                    logs = live.get("recent_logs", [])
                    if logs:
                        lines.append("-" * 40)
                        lines.append("RECENT_LIVE_LOGS:")
                        for entry in logs[-15:]:
                            lines.append(
                                f"  [{entry.get('timestamp')}] {entry.get('level')}: {entry.get('message')}"
                            )

                # Trials List
                trials = data.get("trials", [])
                lines.append("-" * 40)
                lines.append(f"TRIALS ({len(trials)}):")

                known_metric_keys = [
                    "mrr",
                    "best_mrr",
                    "mcc",
                    "auc",
                    "score",
                    "loss",
                    "duration",
                    "hits@1",
                    "hits@3",
                    "hits@10",
                    "accuracy",
                    "f1",
                    "recall",
                    "precision",
                    "ap@10",
                ]

                for t in trials:
                    lines.append(f"  TRIAL #{t.get('id', '??')}:")
                    lines.append(f"    STATE: {t.get('state')}")
                    lines.append(f"    VALUE: {t.get('value')}")

                    params = t.get("params", {})
                    if params:
                        lines.append("    PARAMS:")
                        # Sort params for readability
                        for k in sorted(params.keys()):
                            lines.append(f"      {k}: {params[k]}")

                    # Metrics Handling
                    metrics = t.get("metrics", {}) or {}

                    # Force populate metrics from trial root if missing
                    for k in known_metric_keys:
                        # try exact match
                        if k not in metrics and k in t and t[k] not in (None, ""):
                            metrics[k] = t[k]

                        # try alias (hits1 -> hits@1)
                        alias = k.replace("@", "")
                        if k not in metrics and alias in t and t[alias] not in (None, ""):
                            metrics[k] = t[alias]

                    # If score is still 0 but value is set, use value
                    if metrics.get("score") == 0 and t.get("value"):
                        metrics["score"] = t.get("value")

                    if metrics:
                        lines.append("    METRICS:")
                        # Sort metrics for readability
                        for k in sorted(metrics.keys()):
                            val = metrics[k]
                            if isinstance(val, (float, int)):
                                if isinstance(val, float):
                                    lines.append(f"      {k}: {val:.6f}")
                                else:
                                    lines.append(f"      {k}: {val}")
                            else:
                                lines.append(f"      {k}: {str(val)}")
                    else:
                        lines.append("    METRICS: (none)")

                    lines.append("")

                content = "\n".join(lines).encode("utf-8")
                content_type = "text/plain"
                ext = "txt"

            else:
                raise ValueError(f"Unknown format: {fmt}")

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f'attachment; filename="{filename}.{ext}"')
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Export failed: {e}")
            self._send_json_response({"error": str(e)}, status=500)

    def _serve_status_api(self):
        """Serve a quick status check."""
        status_path = _find_live_status_path()
        try:
            if status_path is not None and status_path.exists():
                data = json.loads(status_path.read_text())
            else:
                data = {
                    "status": "no_data",
                    "message": "HPO not running or no data available",
                }

            self._send_json_response(data)
        except Exception as e:
            self._send_json_response({"error": str(e)}, status=500)

    def _send_json_response(self, data: dict[str, Any], status: int = 200):
        """Send a JSON response."""
        content = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        """Suppress default logging, use our logger instead."""
        logger.debug(f"Dashboard: {args[0]}")


def run_server(port: int = 8766, bind: str = "0.0.0.0"):
    """Run the dashboard server.

    Args:
        port: Port to listen on (default: 8766)
        bind: Address to bind to (default: 0.0.0.0)
    """
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((bind, port), DashboardHandler) as httpd:
        logger.info(f"Dashboard server iniciado em http://{bind}:{port}")
        logger.info(f"Arquivos estáticos: {STATIC_DIR}")
        logger.info(f"Dados HPO: {DATA_CACHE_PATH}")
        print(f"\n  HPO Live Dashboard disponível em: http://localhost:{port}\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Dashboard server encerrado.")


def _find_live_status_path() -> Path | None:
    candidates = [
        settings.OUTPUTS_DIR / "optimization" / "plots" / "live" / "live_status.json",
        settings.OUTPUTS_DIR / "optimization" / "plots" / "live_status.json",
    ]
    # Filter only existing files
    existing = [c for c in candidates if c.exists()]
    if not existing:
        return candidates[0] if candidates else None

    # Priority 1: Check if any file was modified in the last 60 seconds (likely live)
    now = time.time()
    for p in existing:
        if now - p.stat().st_mtime < 60:
            return p

    # Priority 2: Return the freshest one by modification time
    return max(existing, key=lambda p: p.stat().st_mtime)


def _max_updated_at(current: str | None, incoming: str | None) -> str:
    if not incoming:
        return current or datetime.now(timezone.utc).isoformat()
    if not current:
        return incoming
    try:
        current_dt = datetime.fromisoformat(current.replace("Z", "+00:00"))
        incoming_dt = datetime.fromisoformat(incoming.replace("Z", "+00:00"))
        return incoming if incoming_dt >= current_dt else current
    except Exception:
        return incoming


def main():
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(description="HPO Live Dashboard Server")
    parser.add_argument("--port", type=int, default=8766, help="Port to listen on")
    parser.add_argument("--bind", default="0.0.0.0", help="Address to bind to")
    args = parser.parse_args()

    run_server(port=args.port, bind=args.bind)


if __name__ == "__main__":
    # Prevent PFF global signal handlers from blocking simple Ctrl+C exit
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()

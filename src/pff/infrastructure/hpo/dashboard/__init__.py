"""HPO Live Dashboard module.

This module provides a live dashboard for monitoring HPO (Hyperparameter Optimization)
runs. The dashboard is a static React application that reads data from JSON files
written by the HPO callbacks.

Architecture:
    - static/index.html: The React dashboard (fixed design, not generated)
    - The dashboard reads from outputs/.cache/hpo/dashboard_data.json
      (override via live_plots.dashboard_data_path in config/hpo/optimization.yaml)
    - HPO callbacks (LivePlotCallback) write to this JSON file
    - The clean command clears data files but preserves the dashboard HTML

Usage:
    python -m pff.infrastructure.hpo.dashboard.server

    pff dashboard --port 8766
"""

from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent
STATIC_DIR = DASHBOARD_DIR / "static"
DASHBOARD_HTML = STATIC_DIR / "index.html"

__all__ = ["DASHBOARD_DIR", "STATIC_DIR", "DASHBOARD_HTML"]

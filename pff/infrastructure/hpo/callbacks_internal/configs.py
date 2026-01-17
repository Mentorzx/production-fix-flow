"""Shared callback configuration utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from pff.shared.core.config import OPTIMIZATION_CONFIG_PATH
from pff.shared import logger
from pff.shared.core.file_manager import FileManager, ParquetBundle


def _get_callback_config() -> dict[str, Any]:
    """Load callback config from optimization.yaml."""
    fm = FileManager()
    try:
        if fm.exists(OPTIMIZATION_CONFIG_PATH):
            payload = fm.read(OPTIMIZATION_CONFIG_PATH)
            cfg = (
                payload.to_native()
                if isinstance(payload, ParquetBundle)
                else payload or {}
            )
            return cfg.get("callbacks", {}) if isinstance(cfg, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to load callbacks config: {exc}")
    return {}


def _save_matplotlib_figure_png(
    fig: Any,
    output_path: Path,
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> None:
    """Persist a Matplotlib figure to PNG using the utils FileManager.

    Args:
        fig: Matplotlib Figure-compatible object with a `savefig` method.
        output_path: Destination PNG path.
        dpi: Render resolution.
        bbox_inches: Matplotlib bbox mode.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches=bbox_inches)
    FileManager.write_bytes(buffer.getvalue(), output_path)

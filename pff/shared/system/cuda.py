"""CUDA availability helpers.

Design Pattern: Adapter. Wraps PyTorch CUDA detection to avoid noisy warnings in
environments where CUDA calls are present but not supported (e.g., sandboxed
CI/containers).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from pff.shared.core.file_manager import FileManager


def is_cuda_available() -> bool:
    """Check whether CUDA is usable without emitting initialization warnings.

    Returns:
        True if CUDA is available and usable; otherwise False.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip() in {"", "-1", "none", "None"}:
        return False

    fm = FileManager()
    if not any(
        fm.exists(path)
        for path in (
            Path("/dev/nvidiactl"),
            Path("/dev/nvidia0"),
            Path("/dev/dxg"),
        )
    ):
        return False

    try:
        import torch
    except Exception:  # noqa: BLE001
        return False

    if not hasattr(torch, "cuda"):
        return False

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA initialization:.*",
            category=UserWarning,
        )
        try:
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False

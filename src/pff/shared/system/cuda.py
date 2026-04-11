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
from pff.shared.core.logging import logger

_VALID_ACCELERATORS = frozenset({"auto", "cpu", "cuda"})
_CUDA_ALLOCATOR_CONFIG = ",".join(
    [
        "garbage_collection_threshold:0.6",
        "expandable_segments:True",
        "max_split_size_mb:512",
    ]
)


def get_requested_accelerator() -> str:
    """Return the requested accelerator mode from the environment."""
    raw_value = os.environ.get("PFF_ACCELERATOR", "auto").strip().lower()
    if raw_value in _VALID_ACCELERATORS:
        return raw_value
    return "auto"


def _has_visible_cuda_devices() -> bool:
    """Check whether NVIDIA device files are exposed to the process."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip() in {"", "-1", "none", "None"}:
        return False

    fm = FileManager()
    return any(
        fm.exists(path)
        for path in (
            Path("/dev/nvidiactl"),
            Path("/dev/nvidia0"),
            Path("/dev/dxg"),
        )
    )


def _torch_reports_cuda() -> bool:
    """Ask PyTorch whether CUDA can be initialized successfully."""
    try:
        import torch
    except Exception:
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
        except Exception:
            return False


def resolve_accelerator() -> tuple[str, str]:
    """Resolve the effective accelerator and a stable reason string."""
    requested = get_requested_accelerator()
    if requested == "cpu":
        return "cpu", "forced_cpu"

    if not _has_visible_cuda_devices():
        if requested == "cuda":
            return "cpu", "cuda_requested_without_visible_device"
        return "cpu", "no_visible_cuda_device"

    if _torch_reports_cuda():
        return "cuda", "cuda_available"

    if requested == "cuda":
        return "cpu", "cuda_requested_but_torch_unavailable"
    return "cpu", "torch_cuda_unavailable"


def configure_accelerator_environment() -> tuple[str, str]:
    """Apply runtime accelerator environment defaults before CUDA initialization."""
    accelerator, reason = resolve_accelerator()
    requested = get_requested_accelerator()

    if requested == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return accelerator, reason

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _CUDA_ALLOCATOR_CONFIG)
    return accelerator, reason


def log_accelerator_resolution() -> tuple[str, str]:
    """Log the resolved accelerator once during runtime initialization."""
    accelerator, reason = configure_accelerator_environment()
    requested = get_requested_accelerator()
    logger.bind(
        component_name="runtime_accelerator",
        key_parameters={
            "requested": requested,
            "resolved": accelerator,
        },
        stop_reason=reason,
    ).info(
        f"Acelerador resolvido: solicitado={requested}, ativo={accelerator}, motivo={reason}"
    )
    if requested == "cuda" and accelerator != "cuda":
        logger.warning(
            "CUDA requested via PFF_ACCELERATOR=cuda but runtime resolved to CPU."
        )
    return accelerator, reason


def is_cuda_available() -> bool:
    """Check whether CUDA is usable without emitting initialization warnings.

    Returns:
        True if CUDA is available and usable; otherwise False.
    """
    accelerator, _reason = resolve_accelerator()
    return accelerator == "cuda"

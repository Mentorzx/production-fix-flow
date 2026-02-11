import importlib as _importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pff.shared.acceleration.concurrency import ConcurrencyManager, progress_bar
    from pff.shared.acceleration.loop_accelerator import (
        AcceleratorBackend,
        AcceleratorConfig,
        LoopAccelerator,
        accelerate_loop,
    )
    from pff.shared.core.cache import CacheManager, DiskCache
    from pff.shared.core.config_loader import load_config
    from pff.shared.core.file_manager import FileManager
    from pff.shared.core.logging import (
        FORMAT,
        LOG_DIR,
        LogReorderer,
        logger,
        silence_libs,
        timeit,
    )
    from pff.shared.research import Research, TripleStore
    from pff_rust import RuleEncoder, stable_hash

__all__ = [
    "FileManager",
    "CacheManager",
    "load_config",
    "logger",
    "FORMAT",
    "LOG_DIR",
    "LogReorderer",
    "progress_bar",
    "silence_libs",
    "ConcurrencyManager",
    "DiskCache",
    "timeit",
    "LoopAccelerator",
    "AcceleratorConfig",
    "AcceleratorBackend",
    "accelerate_loop",
    "RuleEncoder",
    "Research",
    "TripleStore",
    "stable_hash",
    "core",
    "acceleration",
    "system",
    "ops",
    "global_interrupt_manager",
    "clients",
]

_LAZY_SUBMODULES = {
    "acceleration",
    "clients",
    "core",
    "ops",
    "system",
}

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ConcurrencyManager": ("pff.shared.acceleration.concurrency", "ConcurrencyManager"),
    "progress_bar": ("pff.shared.acceleration.concurrency", "progress_bar"),
    "AcceleratorBackend": (
        "pff.shared.acceleration.loop_accelerator",
        "AcceleratorBackend",
    ),
    "AcceleratorConfig": (
        "pff.shared.acceleration.loop_accelerator",
        "AcceleratorConfig",
    ),
    "LoopAccelerator": ("pff.shared.acceleration.loop_accelerator", "LoopAccelerator"),
    "accelerate_loop": ("pff.shared.acceleration.loop_accelerator", "accelerate_loop"),
    "CacheManager": ("pff.shared.core.cache", "CacheManager"),
    "DiskCache": ("pff.shared.core.cache", "DiskCache"),
    "load_config": ("pff.shared.core.config_loader", "load_config"),
    "FileManager": ("pff.shared.core.file_manager", "FileManager"),
    "FORMAT": ("pff.shared.core.logging", "FORMAT"),
    "LOG_DIR": ("pff.shared.core.logging", "LOG_DIR"),
    "LogReorderer": ("pff.shared.core.logging", "LogReorderer"),
    "logger": ("pff.shared.core.logging", "logger"),
    "silence_libs": ("pff.shared.core.logging", "silence_libs"),
    "timeit": ("pff.shared.core.logging", "timeit"),
    "global_interrupt_manager": ("pff.shared.ops", "global_interrupt_manager"),
    "Research": ("pff.shared.research", "Research"),
    "TripleStore": ("pff.shared.research", "TripleStore"),
    "RuleEncoder": ("pff_rust", "RuleEncoder"),
    "stable_hash": ("pff_rust", "stable_hash"),
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return _importlib.import_module(f"pff.shared.{name}")
    entry = _LAZY_ATTRS.get(name)
    if entry is not None:
        mod = _importlib.import_module(entry[0])
        return getattr(mod, entry[1])
    raise AttributeError(f"module 'pff.shared' has no attribute {name!r}")

# Import logger first (no internal dependencies)
from .core.logger import FORMAT, LOG_DIR, LogReorderer, logger, silence_libs, timeit

# Then concurrency (depends on logger)
from .acceleration.concurrency import (  # noqa: E402
    ConcurrencyManager,
    progress_bar,
)

# Then cache (depends on logger)
from .core.cache import CacheManager, DiskCache  # noqa: E402

# Then file_manager (depends on logger and ConcurrencyManager)
from .core.file_manager import FileManager  # noqa: E402

# Acceleration modules
from .acceleration.loop_accelerator import (  # noqa: E402
    LoopAccelerator,
    AcceleratorConfig,
    AcceleratorBackend,
    accelerate_loop,
)
from .acceleration.symbolic_rule_accelerator import (  # noqa: E402
    SymbolicRuleAccelerator,
    RuleEncoder,
)

# Shared research utilities
from .research import Research, TripleStore  # noqa: E402

# Export submodules for convenience
from . import core  # noqa: E402
from . import acceleration  # noqa: E402
from . import system  # noqa: E402
from . import ops  # noqa: E402
from . import clients  # noqa: E402
from . import hash  # noqa: E402

# Export specific modules for direct import
from .ops import global_interrupt_manager  # noqa: E402
from .acceleration import numba_kernels  # noqa: E402
from .hash import stable_hash  # noqa: E402

__all__ = [
    "FileManager",
    "CacheManager",
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
    "SymbolicRuleAccelerator",
    "RuleEncoder",
    "Research",
    "TripleStore",
    "stable_hash",
    "core",  # noqa: F401
    "acceleration",  # noqa: F401
    "system",  # noqa: F401
    "ops",  # noqa: F401
    "hash",  # noqa: F401
    "global_interrupt_manager",  # noqa: F401
    "numba_kernels",  # noqa: F401
]

__all__ += [
    "clients",
]

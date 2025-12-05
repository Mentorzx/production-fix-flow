# Import logger first (no internal dependencies)
from .core.logger import FORMAT, LOG_DIR, LogReorderer, logger, silence_libs, timeit

# Then concurrency (depends on logger)
from .acceleration.concurrency import (
    ConcurrencyManager,
    first_success,
    progress_bar,
)

# Then cache (depends on logger)
from .core.cache import CacheManager, DiskCache

# Then file_manager (depends on logger and ConcurrencyManager)
from .core.file_manager import FileManager

# Other core modules
from .core.output import ResultCollector
from .core.calibration import ScoreCalibrator

# Acceleration modules
from .acceleration.loop_accelerator import LoopAccelerator, AcceleratorConfig, AcceleratorBackend, accelerate_loop
from .acceleration.symbolic_rule_accelerator import SymbolicRuleAccelerator, RuleEncoder

# Network modules
from .network.endpoints import _ORDER, APIsEndpoints, EndpointFactory

# Dev modules
from .dev.research import Research, TripleStore

# Ops modules
from .ops.cleanup import ShutdownCleanup

# Export submodules for backward compatibility (allows: from pff.utils.module import X)
from . import core
from . import acceleration
from . import system
from . import network
from . import data
from . import ops
from . import dev
from . import clients
from . import hooks
from . import hash
from . import db
from .db import (
    PostgresConfig,
    get_postgres_config,
    notify_postgres,
    register_postgres_listener,
)

#Export specific modules for direct import
from .data import polars_extensions
from .system import hardware_detector, ml_training_profiles, resource_manager
from .data import autofeeding
from .ops import global_interrupt_manager
from .acceleration import numba_kernels
from .explainability import ShapExplainerService, ShapExplainerConfig
from .evaluation.edas import KGEDASEvaluator, EDASResult

# Performance modules
from .performance.training_observer import (
    TrainingObserver,
    TrainingEvent,
    ConsoleObserver,
    MLflowObserver,
    CompositeObserver,
    NullObserver,
    create_default_observer,
)

# ML modules (Strategy, Factory, Template Method patterns)
from .ml import (
    KGEModelStrategy,
    TransEStrategy,
    # TODO[Fase2-DSLFM]: Registrar DSLFMStrategy quando habilitado
    KGEConfig,
    ModelFactory,
    ModelType,
    BaseTrainer,
    TrainerConfig,
)

# Create module aliases for backward compatibility
import sys
sys.modules['pff.utils.logger'] = sys.modules['pff.utils.core.logger']
sys.modules['pff.utils.cache'] = sys.modules['pff.utils.core.cache']
sys.modules['pff.utils.file_manager'] = sys.modules['pff.utils.core.file_manager']
sys.modules['pff.utils.output'] = sys.modules['pff.utils.core.output']
sys.modules['pff.utils.concurrency'] = sys.modules['pff.utils.acceleration.concurrency']
sys.modules['pff.utils.loop_accelerator'] = sys.modules['pff.utils.acceleration.loop_accelerator']
sys.modules['pff.utils.symbolic_rule_accelerator'] = sys.modules['pff.utils.acceleration.symbolic_rule_accelerator']
sys.modules['pff.utils.numba_kernels'] = sys.modules['pff.utils.acceleration.numba_kernels']
sys.modules['pff.utils.endpoints'] = sys.modules['pff.utils.network.endpoints']
sys.modules['pff.utils.research'] = sys.modules['pff.utils.dev.research']
sys.modules['pff.utils.cleanup'] = sys.modules['pff.utils.ops.cleanup']
sys.modules['pff.utils.global_interrupt_manager'] = sys.modules['pff.utils.ops.global_interrupt_manager']
sys.modules['pff.utils.resource_manager'] = sys.modules['pff.utils.system.resource_manager']
sys.modules['pff.utils.autofeeding'] = sys.modules['pff.utils.data.autofeeding']
sys.modules['pff.utils.polars_extensions'] = sys.modules['pff.utils.data.polars_extensions']

__all__ = [
    "FileManager",
    "CacheManager",
    "ScoreCalibrator",
    "APIsEndpoints",
    "Research",
    "logger",
    "ResultCollector",
    "FORMAT",
    "LOG_DIR",
    "LogReorderer",
    "progress_bar",
    "EndpointFactory",
    "_ORDER",
    "silence_libs",
    "ConcurrencyManager",
    "first_success",
    "DiskCache",
    "timeit",
    "ShutdownCleanup",
    "TripleStore",
    "LoopAccelerator",
    "AcceleratorConfig",
    "AcceleratorBackend",
    "accelerate_loop",
    "SymbolicRuleAccelerator",
    "RuleEncoder",
    "ShapExplainerService",
    "ShapExplainerConfig",
    "TrainingObserver",
    "TrainingEvent",
    "ConsoleObserver",
    "MLflowObserver",
    "CompositeObserver",
    "NullObserver",
    "create_default_observer",
    "KGEModelStrategy",
    "TransEStrategy",
    "KGEConfig",
    "ModelFactory",
    "ModelType",
    "BaseTrainer",
    "TrainerConfig",
    ]

__all__ += [
    "db",
    "PostgresConfig",
    "get_postgres_config",
    "notify_postgres",
    "register_postgres_listener",
]

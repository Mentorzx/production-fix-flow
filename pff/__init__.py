import os
import sys
import warnings

try:
    import _xxsubinterpreters

    _xxsubinterpreters_stub = _xxsubinterpreters
except ModuleNotFoundError:
    from pff.shared.compat import (
        xxsubinterpreters_stub as _xxsubinterpreters_stub,
    )

sys.modules.setdefault("_xxsubinterpreters", _xxsubinterpreters_stub)


if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers.utils.hub",
)
import polars as pl  # noqa: E402

from pff.application.services.intelligent_preprocessor import (  # noqa: E402
    IntelligentPreprocessor,
)
from pff.domain.audit.manifest import ManifestParser, TaskModel  # noqa: E402
from pff.drivers.celery.app import celery_app  # noqa: E402
from pff.drivers.orchestrator import Orchestrator  # noqa: E402
from pff.shared.core.config import settings  # noqa: E402

pl.enable_string_cache()

"""
PFF – Production Fix Flow
=========================

Light‑weight orchestrator that executes declarative API sequences on groups of
MSISDNs.  All heavy imports are deferred to sub‑modules so that importing *pff*
never has side‑effects (other than setting up logging).

Public objects
--------------
__version__ : str
    Semantic version string, filled at build time.
"""

__all__ = [
    "__version__",
    "settings",
    "TaskModel",
    "ManifestParser",
    "Orchestrator",
    "IntelligentPreprocessor",
    "celery_app",
]

try:
    from importlib.metadata import version as _version

    __version__: str = _version("pff")
except Exception:
    __version__ = "6.0.0"

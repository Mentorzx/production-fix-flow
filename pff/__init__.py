from importlib.metadata import version as _version
import os
import warnings

# Suppress Transformers deprecation warnings
if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.utils.hub"
)

from .config import settings  # noqa: E402
from .celery_app import celery_app  # noqa: E402
from .manifest import ManifestParser, TaskModel  # noqa: E402
from .orchestrator import Orchestrator  # noqa: E402
from .preprocessor import IntelligentPreprocessor  # noqa: E402

import polars as pl  # noqa: E402

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
    __version__: str = _version("pff")
except Exception:
    __version__ = "6.0.0"

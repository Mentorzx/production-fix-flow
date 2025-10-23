import multiprocessing as mp
from importlib.metadata import version as _version

from .config import settings
from .celery_app import celery_app
from .manifest import ManifestParser, TaskModel
from .orchestrator import Orchestrator
from .preprocessor import IntelligentPreprocessor

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

settings.DATA_DIR.mkdir(exist_ok=True)
settings.OUTPUTS_DIR.mkdir(exist_ok=True)
settings.LOGS_DIR.mkdir(exist_ok=True)

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
    __version__ = "5.0.0"

if mp.current_process().name == "MainProcess":
    print(f" PFF Fênix v{__version__} iniciado. Ambiente configurado.")
    print(f"   - Diretório de Logs: {settings.LOGS_DIR}")
    print(f"   - Diretório de Saída: {settings.OUTPUTS_DIR}")

try:
    from pff.utils.hooks import auto_config  # noqa: F401
except ImportError:
    pass  # Hook opcional

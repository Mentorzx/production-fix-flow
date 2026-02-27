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

from typing import TYPE_CHECKING

__all__ = [
    "__version__",
    "settings",
    "TaskModel",
    "ManifestParser",
    "Orchestrator",
    "IntelligentPreprocessor",
    "celery_app",
]

if TYPE_CHECKING:
    from pff.application.services.intelligent_preprocessor import (
        IntelligentPreprocessor,
    )
    from pff.domain.audit.manifest import ManifestParser, TaskModel
    from pff.drivers.celery.app import celery_app
    from pff.drivers.orchestrator import Orchestrator
    from pff.shared.core.config import settings

try:
    from importlib.metadata import version as _version

    __version__: str = _version("pff")
except Exception:
    __version__ = "6.0.0"


def __getattr__(name: str):
    if name == "settings":
        from pff.shared.core.config import settings

        return settings
    if name == "Orchestrator":
        from pff.drivers.orchestrator import Orchestrator

        return Orchestrator
    if name == "IntelligentPreprocessor":
        from pff.application.services.intelligent_preprocessor import (
            IntelligentPreprocessor,
        )

        return IntelligentPreprocessor
    if name in {"TaskModel", "ManifestParser"}:
        from pff.domain.audit.manifest import ManifestParser, TaskModel

        return TaskModel if name == "TaskModel" else ManifestParser
    if name == "celery_app":
        from pff.drivers.celery.app import celery_app

        return celery_app
    raise AttributeError(f"module 'pff' has no attribute {name}")

"""DSLFM-KGC training manager facade (Template Method/Adapter).

Exposes a stable manager entrypoint while delegating to the existing
`kgc_manager` implementation. Intended to give scripts and validators a
generic path (`pff.domain.learning.dslfm.manager`) without touching internals.
"""

from __future__ import annotations

from .kgc_manager import (  # noqa: F401
    DSLFMKGCManager,
    KGCTrainingConfig,
    train_dslfm_kgc,
)

__all__ = ["DSLFMKGCManager", "KGCTrainingConfig", "train_dslfm_kgc"]

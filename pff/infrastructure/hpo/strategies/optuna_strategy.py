"""Optuna strategy alias (Adapter).

Keeps a generic filename while reusing the existing implementation.
"""

from __future__ import annotations

from .optuna_impl import OptunaStrategy

__all__ = ["OptunaStrategy"]
